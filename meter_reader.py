#!/usr/bin/env python3
"""
meter_reader.py — standalone water meter reader, replacing nohn/watermeter.

Pipeline:
  1. Fetch /raw from overlay-proxy (undistorted, unrotated)
  2. Rotate 244.6°
  3. OCR the 4 digital digit crops (Tesseract)
  4. Detect 4 analog dial angles (spoke sampling, copied from overlay-proxy)
  5. Apply inter-dial gear-lash correction
  6. Assemble reading and apply sanity guards
  7. Output reading to stdout

Usage:
  python3 meter_reader.py                   # live from proxy
  python3 meter_reader.py --image foo.jpg   # offline against stored snapshot
  python3 meter_reader.py --debug           # save annotated debug_reading.jpg
  python3 meter_reader.py --no-guard        # skip threshold / decrease check
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
CAM_SNAPSHOT_URL = os.environ.get("CAM_SNAPSHOT_URL", "http://192.168.x.x/")
HA_URL           = os.environ.get("HA_URL", "")
HA_TOKEN         = os.environ.get("HA_TOKEN", "")
READING_INTERVAL = float(os.environ.get("READING_INTERVAL", "10"))
ROTATE_DEG    = 243.0

# Digital digit crop boxes in rotated-image coordinates.
# Each entry: (x, y, w, h) — ordered most-significant to least-significant.
# Recalibrated 2026-06-03 after camera extension (improved focus/DOF).
DIGITAL_DIGITS = [
    (723, 779, 48, 51),
    (771, 779, 48, 51),
    (819, 779, 48, 51),
    (867, 779, 48, 51),
    (915, 779, 48, 51),
]

# Full strip covering all digital digits; used by the strip OCR path.
_DIGITAL_STRIP = (723, 779, 240, 51)

# Analog dial offsets from strip centre: (dx, dy, r, dark_needle, flip)
# Ordered most-significant → least-significant.
# Recalibrated 2026-06-03 via interactive calibration at ROTATE_DEG=243.0.
_ANALOG_DIAL_OFFSETS = [
    (+267, +138, 65, False, False),  # ×0.1 m³
    (+204, +286, 63, False, False),  # ×0.01 m³
    ( +54, +342, 65, False, False),  # ×0.001 m³
    ( -98, +274, 63, False, False),  # ×0.0001 m³
]

def _make_analog_dials():
    sx, sy, sw, sh = _DIGITAL_STRIP
    cx, cy = sx + sw // 2, sy + sh // 2
    return [(cx + dx, cy + dy, r, dk, fl) for dx, dy, r, dk, fl in _ANALOG_DIAL_OFFSETS]

ANALOG_DIALS = _make_analog_dials()

# Mechanical phase correction for analog dials.
# The gear engagement that drives the next dial happens when the driving dial
# reaches face "9", not face "0".  This means face "0" is actually the first
# graduation PAST the mechanical zero, so all raw digit readings are one digit
# low.  Applied as a decimal addition (0.1111) so carry propagates correctly:
# e.g. raw 0.7491 + 0.1111 = 0.8602, not the broken per-digit 0.8502.
DIAL_PHASE_CORRECTION = 0.1111

# Gear-lash correction thresholds.
# LASH_HIGH: more-significant dial frac above this → approaching next digit, snap up.
# LASH_LOW:  more-significant dial frac below this → just crossed digit boundary, snap up.
# LASH_EXT_DEG: pass-1 trigger window — how many degrees past 0° the less-significant
# dial may be while still considered "just completed a revolution". Covers digit 0
# (36°) plus buffer for fast-flow frames where the dial advances significantly between
# snapshots before gear lash resolves mechanically.
# LASH_NEAR_ZERO: frac in digit 9 at which the dial is treated as effectively 0.
LASH_HIGH      = 0.60
LASH_LOW       = 0.20
LASH_EXT_DEG   = 120.0  # ~3.3 digits past 0° crossing
LASH_NEAR_ZERO = 0.90

MAX_STEP       = float(os.environ.get("MAX_STEP", 0.05))  # m³ per sample
ALLOW_DECREASE = False

STATE_FILE    = Path(os.environ.get("STATE_FILE",
                    str(Path(__file__).parent / ".meter_state.json")))
INITIAL_VALUE = float(os.environ["INITIAL_VALUE"]) if "INITIAL_VALUE" in os.environ else None
FLOW_MAX_AGE  = float(os.environ["FLOW_MAX_AGE"]) if "FLOW_MAX_AGE" in os.environ else None  # seconds

log = logging.getLogger(__name__)


# ── Image helpers ──────────────────────────────────────────────────────────────
def rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    """Expand-canvas rotation matching Imagick rotateImage behaviour."""
    h, w = img.shape[:2]
    rad = np.radians(deg)
    cos_a, sin_a = abs(np.cos(rad)), abs(np.sin(rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


# ── Digital OCR ────────────────────────────────────────────────────────────────
def _ocr_single_digit(crop: np.ndarray) -> int | None:
    """
    Return int 0–9 from a digit crop, or None on failure.
    Tries both normal and inverted threshold; picks the highest-confidence result.
    """
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)

    cfg = r"--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"

    for invert in (False, True):
        src = 255 - gray if invert else gray
        _, thresh = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Pad so Tesseract doesn't clip characters at the edge
        thresh = cv2.copyMakeBorder(thresh, 12, 12, 12, 12,
                                    cv2.BORDER_CONSTANT, value=255)
        data = pytesseract.image_to_data(
            thresh, config=cfg, output_type=pytesseract.Output.DICT)
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            if text.isdigit():
                return int(text)

    return None


def read_digital_digits(img: np.ndarray,
                        last_int: int | None = None) -> list[int | None]:
    """Return list of 5 ints (or None per failed digit) from the digital counter.

    Primary path: reads the full digit strip as one unit with PSM 8 — more
    reliable than per-digit PSM 10 for this meter's font.  Falls back to
    per-digit OCR if the strip result doesn't yield exactly 5 digits, or if
    the zfilled result is implausibly far from last_int (e.g. Tesseract dropped
    a leading zero AND misread another — "00297" → "9297" → zfill "09297").
    """
    sx, sy, sw, sh = _DIGITAL_STRIP
    strip = img[sy:sy + sh, sx:sx + sw]
    strip3x = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(cv2.cvtColor(strip3x, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.copyMakeBorder(thresh, 12, 12, 12, 12,
                                cv2.BORDER_CONSTANT, value=255)
    n = len(DIGITAL_DIGITS)
    for psm in (7, 6, 8):
        cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
        result = "".join(pytesseract.image_to_string(thresh, config=cfg).split())
        # Accept up to 2 missing leading zeros — the counter left-pads them and
        # Tesseract often drops them from the strip; zfill restores them.
        # Internal spaces are removed above so "0 2 5" → "025" before the check.
        if result.isdigit() and n - 2 <= len(result) <= n:
            digits = [int(c) for c in result.zfill(n)]
            if last_int is None or abs(int("".join(str(d) for d in digits)) - last_int) <= 5:
                return digits
            log.warning("strip OCR '%s' (zfilled '%s') looks wrong vs last=%d — falling back",
                        result, result.zfill(n), last_int)
            break

    log.warning("strip OCR gave no clean 5-digit result — falling back to per-digit")
    return [_ocr_single_digit(img[y:y + h, x:x + w])
            for x, y, w, h in DIGITAL_DIGITS]


# ── Analog dial detection ──────────────────────────────────────────────────────
def detect_needle_angle(img_bgr: np.ndarray, cx: int, cy: int, r: int,
                        dark_needle: bool = False) -> float | None:
    """
    Spoke-sampled needle angle (0° = 12-o'clock, clockwise).
    Returns degrees 0–359 or None if no clear needle found.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    if dark_needle:
        gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
        signal  = np.clip(110 - gray, 0, 255)
        r_outer = r * 0.85
    else:
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   50, 30]), np.array([20, 255, 255])),
            cv2.inRange(hsv, np.array([155, 50, 30]), np.array([180, 255, 255])),
        )
        signal  = np.where(red_mask > 0, hsv[:, :, 1].astype(float), 0.0)
        r_outer = float(r)

    hub_r  = r * 0.22
    n_samp = 48

    angles_rad = np.radians(np.arange(360))
    t_vals     = hub_r + (r_outer - hub_r) * np.arange(n_samp) / n_samp
    H, W       = img_bgr.shape[:2]
    px = np.clip((cx + np.outer(np.sin(angles_rad), t_vals)).astype(int), 0, W - 1)
    py = np.clip((cy - np.outer(np.cos(angles_rad), t_vals)).astype(int), 0, H - 1)

    scores = (signal[py, px] * t_vals).sum(axis=1)
    if scores.max() < 200:
        return None

    k      = np.ones(9) / 9
    scores = np.convolve(np.tile(scores, 3), k, mode="same")[360:720]
    return float(np.argmax(scores))


def read_analog_dials(img: np.ndarray) -> list[float | None]:
    """Return list of 4 raw angles (degrees) — most→least significant."""
    angles = []
    for cx, cy, r, dark, flip in ANALOG_DIALS:
        angle = detect_needle_angle(img, cx, cy, r, dark_needle=dark)
        if angle is not None and flip:
            angle = (angle + 180) % 360
        angles.append(angle)
    return angles


# ── Inter-dial gear-lash correction ───────────────────────────────────────────
def correct_gear_lash(angles: list[float | None]) -> list[float | None]:
    """
    Two-pass bottom-up gear-lash correction.

    Pass 1 (normal): when less-significant dial is at digit 0, the more-significant
    dial may lag behind due to mechanical gear lash — snap it up if in the LASH zone.

    Pass 2 (near-zero): when less-significant dial is at digit 9 with frac >=
    LASH_NEAR_ZERO (essentially completed its revolution but detection noise keeps
    it just below 0°), snap it to 0° and apply the same gear-lash correction.
    """
    result = list(angles)

    # Pass 1 — normal digit-0 trigger.
    # Trigger is checked against the ORIGINAL angles so that snapping a less-
    # significant dial in one iteration does not suppress the trigger for a
    # more-significant dial that also needs correction.
    for i in range(len(result) - 2, -1, -1):
        if angles[i] is None or angles[i + 1] is None:
            continue
        frac_next  = (angles[i + 1] / 36.0) % 1.0
        frac_cur   = (result[i]     / 36.0) % 1.0
        # Trigger: next dial is within LASH_EXT_DEG of the 0° crossing.
        # This covers all of digit 0 plus a buffer into digit 1/2 where gear
        # lash may still be unresolved (fast flow = next dial advances far
        # between snapshots before lash resolves mechanically).
        # Exception: if the current dial is a near-zero digit 9, pass 2 owns
        # it — pass 2 does the full cascade snap including the parent dial.
        cur_digit      = int(result[i]      / 36.0) % 10
        pass2_owns     = (cur_digit == 9 and frac_cur >= LASH_NEAR_ZERO)
        angle_next_mod = angles[i + 1] % 360.0
        if not (angle_next_mod < LASH_EXT_DEG and not pass2_owns):
            continue
        # LASH_LOW only applies in the core digit-0 zone: in the extended buffer
        # (next dial past 36°) low frac is legitimate — the current dial may
        # have just recently crossed its own boundary.
        in_core_zone = angle_next_mod < 36.0
        if frac_cur > LASH_HIGH or (in_core_zone and frac_cur < LASH_LOW):
            snapped = (int(result[i] / 36.0) + 1) % 10
            old     = result[i]
            result[i] = float(snapped * 36)
            log.info("gear-lash: dial %d %.1f°(frac %.2f) → digit %d  "
                     "[dial %d at %.1f°, %.0f° past 0]",
                     i + 1, old, frac_cur, snapped, i + 2,
                     angles[i + 1], angle_next_mod)

    # Pass 2 — near-zero snap: treat digit 9 at frac >= LASH_NEAR_ZERO as digit 0
    for i in range(len(result) - 2, -1, -1):
        if result[i] is None or result[i + 1] is None:
            continue
        pos_next   = result[i + 1] / 36.0
        digit_next = int(pos_next) % 10
        frac_next  = pos_next % 1.0
        if digit_next != 9 or frac_next < LASH_NEAR_ZERO:
            continue
        old_next      = result[i + 1]
        result[i + 1] = 0.0          # snap less-significant dial to 0°
        frac_cur      = (result[i] / 36.0) % 1.0
        if frac_cur > LASH_HIGH or frac_cur < LASH_LOW:
            snapped    = (int(result[i] / 36.0) + 1) % 10
            old        = result[i]
            result[i]  = float(snapped * 36)
            log.info("gear-lash near-zero: dial %d %.1f°(frac %.2f) → digit %d  "
                     "[dial %d %.1f°→0°]",
                     i + 1, old, frac_cur, snapped, i + 2, old_next)
        else:
            log.info("gear-lash near-zero: dial %d snapped to 0°  "
                     "[dial %d frac %.2f not in lash zone]",
                     i + 2, old_next, i + 1, frac_cur)

    return result


# ── Reading assembly ───────────────────────────────────────────────────────────
def angle_to_digit(angle: float) -> int:
    return int(angle / 36.0) % 10


def assemble_reading(digital: list[int | None],
                     analog_angles: list[float | None]) -> float:
    if any(d is None for d in digital):
        raise ValueError(f"OCR failure — digital digits: {digital}")
    if any(a is None for a in analog_angles):
        raise ValueError(f"Needle detection failure — angles: {analog_angles}")

    integer_part = int("".join(str(d) for d in digital))

    def _round_last(a: float) -> int:
        d = int(a / 36.0) % 10
        return d if d == 9 or (a / 36.0) % 1.0 < 0.5 else (d + 1) % 10

    def _round_intermediate(a_parent: float, a_child: float) -> int:
        # Round-to-nearest only when within ~4.5° of the next-digit boundary
        # (frac >= 7/8). The child's absolute angle cannot confirm the parent's
        # fractional position without a meter-specific phase-offset calibration,
        # so only the parent frac is used; the child angle is reserved for the
        # gear-lash correction pass that runs before this function.
        pos  = a_parent / 36.0
        d    = int(pos) % 10
        frac = pos % 1.0
        if d == 9 or frac < 0.875:
            return d
        return (d + 1) % 10

    analog_digits = []
    for i, a in enumerate(analog_angles):
        if i < len(analog_angles) - 1:
            analog_digits.append(_round_intermediate(a, analog_angles[i + 1]))
        else:
            analog_digits.append(_round_last(a))
    fractional = sum(d * 10 ** -(i + 1) for i, d in enumerate(analog_digits))
    fractional = round(fractional + DIAL_PHASE_CORRECTION, 4) % 1.0
    return round(integer_part + fractional, 4)


# ── Rollover calibration and disambiguation ────────────────────────────────────
#
# At every integer-digit rollover (N→N+1 m³), all four analog dials return to
# their mechanical zero simultaneously.  Recording each dial's raw angle at that
# moment gives its zero offset.  Accuracy degrades for faster dials because the
# 19-minute drum-transition window lets them advance:
#   A0 (×0.1)  : ~0.4° error  → reliable
#   A1 (×0.01) : ~3.6° error  → usable
#   A2 (×0.001): ~36°  error  → poor, skip
#   A3 (×0.0001): ~360° error → skip entirely
#
# Once A0 and A1 are calibrated, they resolve OCR ambiguity during the slow
# mechanical transition of the units (pos4) and tens (pos3) digits respectively.

# dial index → digital position it drives
_DIAL_DRIVES_POS = {0: 4, 1: 3, 2: 2, 3: 1}
# digital position → dial index that drives it
_POS_DRIVEN_BY_DIAL = {v: k for k, v in _DIAL_DRIVES_POS.items()}
# Dials reliable enough to calibrate and use for disambiguation
_CALIBRATE_DIALS = (0, 1)

# Fraction of a revolution within which we declare "just crossed zero" or
# "about to cross zero" for rollover disambiguation.
_ROLLOVER_BAND = 0.15


def _dial_fraction(raw_angle: float | None, zero_offset: float | None) -> float | None:
    """Fractional position 0.0–1.0 relative to calibrated zero. None if uncalibrated."""
    if raw_angle is None or zero_offset is None:
        return None
    return ((raw_angle - zero_offset) % 360) / 360.0


def calibrate_from_rollover(digital: list[int | None],
                            angles: list[float | None],
                            state: dict) -> dict:
    """
    Called after a reading is accepted. If the integer part incremented by 1,
    record dial angles as zero offsets.  Returns updated state (not yet saved).
    """
    last = state.get("last_reading")
    if last is None or any(d is None for d in digital):
        return state

    last_int = int(last)
    new_int  = int("".join(str(d) for d in digital))

    if new_int != last_int + 1:
        return state

    # The first clean new-digit frame is captured when the digit drum has just
    # become readable but may not yet be fully settled.  The mechanical carry
    # completes one dial graduation (36°) later — at the face "1" position rather
    # than face "0".  Apply a fixed +36° phase correction to align the stored
    # zero with the true mechanical completion point.
    _CARRY_PHASE_DEG = 36.0

    offsets = state.setdefault("dial_zero_offsets", [None] * 4)
    for dial_idx in _CALIBRATE_DIALS:
        if dial_idx < len(angles) and angles[dial_idx] is not None:
            corrected = (angles[dial_idx] + _CARRY_PHASE_DEG) % 360
            offsets[dial_idx] = corrected
            log.info("dial calibration: A%d (×%s) zero=%.1f°  (%d→%d rollover, +%.0f° phase)",
                     dial_idx, ["0.1", "0.01", "0.001", "0.0001"][dial_idx],
                     corrected, last_int, new_int, _CARRY_PHASE_DEG)

    return state


def resolve_rollover(digital: list[int | None],
                     angles: list[float | None],
                     state: dict) -> list[int | None]:
    """
    Use the calibrated dial fraction to correct digits that OCR got wrong during
    a mechanical drum transition.  Operates on any OCR value (including None and
    garbled reads such as 9 during a 7→8 transition) whenever the dial fraction
    is inside the rollover band:
      fraction < _ROLLOVER_BAND  → dial just crossed zero → force expected_new
      fraction > 1-_ROLLOVER_BAND → dial about to cross zero → force expected_old
    Outside the rollover band the digit is left as-is.
    """
    last = state.get("last_reading")
    if last is None:
        return digital

    offsets   = state.get("dial_zero_offsets", [None] * 4)
    last_digs = [int(c) for c in f"{int(last):05d}"]
    result    = list(digital)

    for pos, dial_idx in _POS_DRIVEN_BY_DIAL.items():
        if dial_idx not in _CALIBRATE_DIALS:
            continue

        frac = _dial_fraction(
            angles[dial_idx] if dial_idx < len(angles) else None,
            offsets[dial_idx])
        if frac is None:
            continue

        d            = result[pos]
        expected_old = last_digs[pos]
        expected_new = (expected_old + 1) % 10

        if frac < _ROLLOVER_BAND and d != expected_new:
            result[pos] = expected_new
            log.info("rollover assist: pos%d OCR=%s → %d  (A%d frac=%.3f, past zero)",
                     pos, d, expected_new, dial_idx, frac)
        elif frac > 1 - _ROLLOVER_BAND and d != expected_old:
            result[pos] = expected_old
            log.info("rollover assist: pos%d OCR=%s → %d  (A%d frac=%.3f, before zero)",
                     pos, d, expected_old, dial_idx, frac)

    return result


# ── Sanity guards ──────────────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        if INITIAL_VALUE is not None:
            return {"last_reading": INITIAL_VALUE}
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def validate(new_val: float, state: dict) -> tuple[bool, str]:
    last = state.get("last_reading")
    if last is None:
        return True, "first reading"
    delta = new_val - last
    if abs(delta) > MAX_STEP:
        return False, f"jump {delta:+.4f} exceeds MAX_STEP {MAX_STEP}"
    if not ALLOW_DECREASE and delta < 0:
        return False, f"decrease not allowed ({last:.4f} → {new_val:.4f})"
    return True, "ok"


# ── Debug annotation ───────────────────────────────────────────────────────────
def annotate(img: np.ndarray, digital: list[int | None],
             angles_raw: list[float | None],
             angles_cor: list[float | None]) -> np.ndarray:
    out = img.copy()
    for i, (x, y, w, h) in enumerate(DIGITAL_DIGITS):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
        if digital[i] is not None:
            cv2.putText(out, str(digital[i]), (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    for i, (cx, cy, r, _, _) in enumerate(ANALOG_DIALS):
        cv2.circle(out, (cx, cy), r, (255, 160, 0), 2)
        for angle, color in ((angles_raw[i], (120, 120, 120)),
                              (angles_cor[i], (0, 0, 255))):
            if angle is not None:
                a = math.radians(angle)
                tip = (int(cx + (r - 12) * math.sin(a)),
                       int(cy - (r - 12) * math.cos(a)))
                cv2.line(out, (cx, cy), tip, color, 2)
        d = angle_to_digit(angles_cor[i]) if angles_cor[i] is not None else "?"
        cv2.putText(out, str(d), (cx - 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return out


# ── Home Assistant integration ─────────────────────────────────────────────────
def push_to_ha(reading: float, flow_lpm: float | None = None) -> None:
    if not HA_URL or not HA_TOKEN:
        return
    headers = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    entities = [
        (
            "sensor.water_meter",
            f"{reading:.4f}",
            {
                "unit_of_measurement": "m³",
                "device_class": "water",
                "state_class": "total_increasing",
                "friendly_name": "Water Meter",
            },
        ),
    ]
    if flow_lpm is not None:
        entities.append((
            "sensor.water_meter_flow",
            f"{flow_lpm:.3f}",
            {
                "unit_of_measurement": "L/min",
                "device_class": "volume_flow_rate",
                "state_class": "measurement",
                "friendly_name": "Water Meter Flow",
            },
        ))
    for entity_id, state, attrs in entities:
        try:
            resp = requests.post(
                f"{HA_URL}/api/states/{entity_id}",
                json={"state": state, "attributes": attrs},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("HA push failed (%s): %s", entity_id, e)


# ── Main ───────────────────────────────────────────────────────────────────────
def process(img: np.ndarray, debug: bool = False,
            state: dict | None = None) -> tuple[float, list, list]:
    """
    Returns (reading, digital, angles_cor).
    Pass `state` to enable rollover disambiguation using calibrated dial zeros.
    """
    rotated  = rotate_image(img, ROTATE_DEG)
    last_int = int(state["last_reading"]) if state and "last_reading" in state else None
    digital  = read_digital_digits(rotated, last_int=last_int)
    angles_raw = read_analog_dials(rotated)
    angles_cor = correct_gear_lash(angles_raw)

    if state:
        digital = resolve_rollover(digital, angles_cor, state)

    reading = assemble_reading(digital, angles_cor)

    log.info("digital=%s  raw_angles=%s  corrected_digits=%s  reading=%.4f",
             digital,
             [f"{a:.1f}" if a is not None else "None" for a in angles_raw],
             [angle_to_digit(a) if a is not None else "?" for a in angles_cor],
             reading)

    if debug:
        ann = annotate(rotated, digital, angles_raw, angles_cor)
        cv2.imwrite("debug_reading.jpg", ann)
        log.info("Annotated image saved to debug_reading.jpg")

    return reading, digital, angles_cor


def _fetch_image(image_path: str | None) -> np.ndarray:
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        return img
    r = requests.get(CAM_SNAPSHOT_URL, timeout=15)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from camera")
    return img


def _run_once(image_path: str | None, debug: bool, no_guard: bool) -> float | None:
    """Fetch, process, validate, push. Returns accepted reading or None."""
    state = load_state()
    try:
        img = _fetch_image(image_path)
        reading, digital, angles_cor = process(img, debug=debug, state=state)
    except (ValueError, RuntimeError, requests.exceptions.RequestException) as e:
        log.error("%s", e)
        return None

    if no_guard:
        print(f"{reading:.4f}")
        push_to_ha(reading)
        return reading

    ok, reason = validate(reading, state)
    if not ok:
        log.warning("Rejected: %s", reason)
        return None

    now = time.time()
    last_ts = state.get("last_reading_ts")
    last_val = state.get("last_reading")
    flow_lpm = None
    if last_val is not None and last_ts is not None:
        elapsed = now - last_ts
        if elapsed <= 0:
            log.warning("flow rate skipped: elapsed=%.3fs (clock went backwards?)", elapsed)
        elif FLOW_MAX_AGE is not None and elapsed > FLOW_MAX_AGE:
            log.info("flow rate skipped: elapsed=%.1fs exceeds FLOW_MAX_AGE=%.1fs", elapsed, FLOW_MAX_AGE)
        else:
            flow_lpm = (reading - last_val) * 1000 / elapsed * 60

    state = calibrate_from_rollover(digital, angles_cor, state)
    state["last_reading"] = reading
    state["last_reading_ts"] = now
    save_state(state)
    print(f"{reading:.4f}", flush=True)
    push_to_ha(reading, flow_lpm)
    return reading


def main() -> None:
    ap = argparse.ArgumentParser(description="Read water meter from camera.")
    ap.add_argument("--image",    help="Load from file instead of live camera")
    ap.add_argument("--debug",    action="store_true", help="Write debug_reading.jpg")
    ap.add_argument("--no-guard", action="store_true", help="Skip sanity guards")
    ap.add_argument("--loop",     action="store_true",
                    help="Run continuously on a fixed interval (READING_INTERVAL env var)")
    ap.add_argument("--interval", type=float, default=None,
                    help="Override READING_INTERVAL (seconds, default 10)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if not args.loop:
        result = _run_once(args.image, args.debug, args.no_guard)
        sys.exit(0 if result is not None else 1)

    interval = args.interval if args.interval is not None else READING_INTERVAL
    log.info("Loop mode: interval=%.1fs  cam=%s  ha=%s",
             interval, CAM_SNAPSHOT_URL, HA_URL or "(not configured)")

    while True:
        t0 = time.monotonic()
        _run_once(args.image, args.debug, args.no_guard)
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


if __name__ == "__main__":
    main()
