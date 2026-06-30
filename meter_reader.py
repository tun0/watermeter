#!/usr/bin/env python3
"""
meter_reader.py — water meter reader.

Pipeline:
  1. Fetch image from camera (or load from file)
  2. Rotate by configured degrees
  3. OCR the 5 digital digit crops (Tesseract)
  4. Detect 4 analog dial angles (spoke sampling)
  5. Apply dial influence correction (boundary disambiguation via gear cascade)
  6. Apply rollover coverage (override OCR during digit drum transition)
  7. Assemble reading from corrected digits
  8. Validate against previous reading
  9. Push to Home Assistant

Usage:
  python3 meter_reader.py                         # live from camera
  python3 meter_reader.py --image foo.jpg         # offline against stored snapshot
  python3 meter_reader.py --debug                 # save annotated debug_reading.jpg
  python3 meter_reader.py --no-guard              # skip validation guards
  python3 meter_reader.py --loop                  # run continuously (pushes to HA)
  python3 meter_reader.py --loop --interval 30    # override interval (seconds)
  python3 meter_reader.py --last-reading 307.2500 # force last_reading baseline (once, no --loop)
  python3 meter_reader.py --push                  # one-off run that also pushes to HA
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
# .env.dist (co-located with this file) is loaded as a defaults layer on startup.
# Env vars already set (e.g. from a configmap or .env) take precedence.
# Required variables not covered by .env.dist or the environment cause an
# explicit failure at startup.

def _load_env_dist() -> None:
    env_dist = Path(__file__).parent / ".env.dist"
    if not env_dist.exists():
        return
    for line in env_dist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, value = line.partition('=')
        os.environ.setdefault(key.strip(), value.strip())

_load_env_dist()


def _env(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(
            f"Required environment variable {name!r} is not set (see .env.dist)")
    return val


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',')]


def _parse_crop(s: str) -> tuple[int, int, int, int]:
    x, y, w, h = (int(v.strip()) for v in s.split(','))
    return x, y, w, h


def _parse_crop_list(s: str) -> list[tuple[int, int, int, int]]:
    return [_parse_crop(rec) for rec in s.split(';')]


def _parse_dial_list(s: str) -> list[tuple[int, int, int]]:
    result = []
    for rec in s.split(';'):
        parts = [p.strip() for p in rec.split(',')]
        result.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return result


# Integration (site-specific; no defaults)
CAM_SNAPSHOT_URL = _env("CAM_SNAPSHOT_URL")
HA_URL           = _env("HA_URL")
HA_TOKEN         = _env("HA_TOKEN")

# Camera / image geometry (camera-specific calibration)
ROTATE_DEG     = float(_env("ROTATE_DEG"))
DIGITAL_STRIP = _parse_crop(_env("DIGITAL_STRIP"))
DIGITAL_DIGITS = _parse_crop_list(_env("DIGITAL_DIGITS"))
ANALOG_DIALS   = _parse_dial_list(_env("ANALOG_DIALS"))

# Dial calibration — physical zero offsets (degrees), order A0..A3
DIAL_ZERO_OFFSETS = _parse_floats(_env("DIAL_ZERO_OFFSETS"))

# Dial influence thresholds (fraction 0.0–1.0)
DIAL_INFLUENCE_HIGH = float(_env("DIAL_INFLUENCE_HIGH"))
DIAL_INFLUENCE_LOW  = float(_env("DIAL_INFLUENCE_LOW"))

# Rollover: corrected_fraction >= ROLLOVER_START means digit drum is transitioning
ROLLOVER_START = float(_env("ROLLOVER_START"))

# Rate safeguards
READING_INTERVAL = float(_env("READING_INTERVAL"))
MAX_STEP         = float(_env("MAX_STEP"))
MAX_DELTA_CAP    = float(_env("MAX_DELTA_CAP"))
JITTER_TOLERANCE = float(_env("JITTER_TOLERANCE"))

# Snapshots (optional — disabled when SNAPSHOT_DIR is unset)
SNAPSHOT_DIR          = os.environ.get("SNAPSHOT_DIR")
_max_age              = os.environ.get("SNAPSHOT_MAX_AGE_DAYS")
SNAPSHOT_MAX_AGE_DAYS = float(_max_age) if _max_age is not None else None

# State
STATE_FILE   = Path(_env("STATE_FILE"))
FLOW_MAX_AGE = float(os.environ["FLOW_MAX_AGE"]) if "FLOW_MAX_AGE" in os.environ else None

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
    """Return int 0–9 from a digit crop, or None on failure."""
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # CLAHE normalises contrast before blur+threshold — helps thin digits in uneven light
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    cfg = r"--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789"
    best_digit, best_conf = None, -1
    for invert in (False, True):
        src = 255 - gray if invert else gray
        for thresh in (
            cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2),
        ):
            padded = cv2.copyMakeBorder(thresh, 12, 12, 12, 12,
                                        cv2.BORDER_CONSTANT, value=255)
            data = pytesseract.image_to_data(
                padded, config=cfg, output_type=pytesseract.Output.DICT)
            for text, conf in zip(data["text"], data["conf"]):
                text = text.strip()
                if text.isdigit() and int(conf) > best_conf:
                    best_digit, best_conf = int(text), int(conf)
    return best_digit


def read_digital_digits(img: np.ndarray, last_int: int | None = None,
                        pinned: dict[int, int] | None = None) -> list[int | None]:
    """Return list of 5 ints (or None per failed digit) from the digital counter.

    pinned: positions already determined by rollover logic. When non-empty, strip
    OCR is skipped entirely (it fails on the rotating drum anyway) and those
    positions are not OCR'd — their values are taken directly from the dict.
    """
    if pinned is None:
        pinned = {}
    n = len(DIGITAL_DIGITS)

    if not pinned:
        sx, sy, sw, sh = DIGITAL_STRIP
        strip = img[sy:sy + sh, sx:sx + sw]
        strip3x = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(cv2.cvtColor(strip3x, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Dilated variant thickens thin strokes (like "1") to reduce Tesseract drop-outs
        thresh_dilated = cv2.dilate(thresh, np.ones((3, 1), np.uint8))

        best_len, best_digits = 0, None
        seen: dict[str, str] = {}   # raw_result → rejection reason (for diagnostics)
        for t_label, t in (("plain", thresh), ("dilated", thresh_dilated)):
            padded = cv2.copyMakeBorder(t, 12, 12, 12, 12,
                                        cv2.BORDER_CONSTANT, value=255)
            for psm in (7, 6, 8):
                cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
                result = "".join(pytesseract.image_to_string(padded, config=cfg).split())
                if not result.isdigit():
                    seen.setdefault(repr(result), f"non-digit (psm={psm},{t_label})")
                    continue
                rlen = len(result)
                # Without a last_int anchor, short zfills are untrustworthy — need full n digits
                min_len = n if last_int is None else n - 2
                if not (min_len <= rlen <= n):
                    seen.setdefault(repr(result),
                                    f"len={rlen} outside [{min_len},{n}] (psm={psm},{t_label})")
                    continue
                digits = [int(c) for c in result.zfill(n)]
                assembled_int = int("".join(str(d) for d in digits))
                if last_int is not None and not (0 <= assembled_int - last_int <= 1):
                    seen.setdefault(repr(result),
                                    f"assembled={assembled_int} implausible vs last={last_int}"
                                    f" (psm={psm},{t_label})")
                    continue
                # Prefer longer results (exact n-digit match is more reliable than zfilled shorter)
                if rlen > best_len:
                    best_len, best_digits = rlen, digits
                    seen[repr(result)] = "accepted"
                if rlen == n:
                    return best_digits  # perfect — no point trying more variants

        if best_digits is not None:
            return best_digits
        attempts = "; ".join(f"{k}: {v}" for k, v in seen.items()) or "all empty"
        log.info("strip OCR fallback to per-digit — attempts: %s", attempts)

    digits: list[int | None] = [pinned.get(i) for i in range(n)]
    for i, (x, y, w, h) in enumerate(DIGITAL_DIGITS):
        if i not in pinned:
            digits[i] = _ocr_single_digit(img[y:y + h, x:x + w])

    if last_int is not None:
        last_digs = [int(c) for c in f"{last_int:05d}"]
        for i, d in enumerate(digits):
            if i in pinned:
                continue
            rollover = (last_digs[i] + 1) % 10
            if d is None:
                digits[i] = last_digs[i]
                log.debug("digit[%d] OCR failed — using last known %d", i, last_digs[i])
            elif d != last_digs[i] and d != rollover:
                log.debug("digit[%d] OCR=%d implausible (last=%d, rollover=%d) — using last",
                          i, d, last_digs[i], rollover)
                digits[i] = last_digs[i]
        assembled = int("".join(str(d) for d in digits))
        if not (0 <= assembled - last_int <= 1):
            log.info("per-digit assembled %d implausible vs last=%d — reverting",
                     assembled, last_int)
            digits = [pinned.get(i, last_digs[i]) for i in range(n)]
    return digits


# ── Analog dial detection ──────────────────────────────────────────────────────
def detect_needle_angle(img_bgr: np.ndarray, cx: int, cy: int, r: int) -> float | None:
    """Spoke-sampled needle angle (0° = 12-o'clock, clockwise). Returns 0–359 or None."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
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
    """Return list of 4 raw angles (degrees) — A0..A3, most→least significant."""
    return [detect_needle_angle(img, cx, cy, r) for cx, cy, r in ANALOG_DIALS]


# ── Corrected angle primitives ─────────────────────────────────────────────────

def corrected_angle(raw: float, zero_offset: float) -> float:
    """Raw angle adjusted to the dial's calibrated zero position."""
    return (raw - zero_offset) % 360.0


def corrected_digit(corr: float) -> int:
    """Integer digit 0–9 from a corrected angle."""
    return int(corr / 36.0) % 10


# ── Dial influence correction ──────────────────────────────────────────────────

def dial_influenced_digit(n: int, raw_angles: list[float | None]) -> int:
    """
    Digit for dial n, with boundary ambiguity resolved using dial n+1.

    Reads as (k+1) % 10 only when BOTH:
      sub_frac(n)          > DIAL_INFLUENCE_HIGH  — n is near its upper boundary
      corrected(n+1) / 360 < DIAL_INFLUENCE_LOW   — n+1 just passed 0 (just drove n)

    Reads as (k-1) % 10 only when BOTH:
      sub_frac(n)          < DIAL_INFLUENCE_LOW   — n just crossed a boundary
      corrected(n+1) / 360 > DIAL_INFLUENCE_HIGH  — n+1 hasn't completed its revolution
    """
    raw = raw_angles[n]
    if raw is None:
        return 0

    corr = corrected_angle(raw, DIAL_ZERO_OFFSETS[n])
    pos  = corr / 36.0
    k    = int(pos) % 10
    sub  = pos % 1.0

    if n + 1 >= len(raw_angles) or raw_angles[n + 1] is None:
        return k

    driver_corr = corrected_angle(raw_angles[n + 1], DIAL_ZERO_OFFSETS[n + 1])
    driver_sub  = driver_corr / 360.0

    if sub > DIAL_INFLUENCE_HIGH and driver_sub < DIAL_INFLUENCE_LOW:
        return (k + 1) % 10
    if sub < DIAL_INFLUENCE_LOW and driver_sub > DIAL_INFLUENCE_HIGH:
        return (k - 1 + 10) % 10
    return k


# ── Reading assembly ───────────────────────────────────────────────────────────

def assemble_reading(digital: list[int | None],
                     raw_angles: list[float | None]) -> float:
    if any(d is None for d in digital):
        raise ValueError(f"OCR failure — digital digits: {digital}")
    if any(a is None for a in raw_angles):
        raise ValueError(f"Needle detection failure — angles: {raw_angles}")

    integer_part = int("".join(str(d) for d in digital))
    n            = len(raw_angles)
    influenced   = [dial_influenced_digit(i, raw_angles) for i in range(n)]
    raw_corr     = [corrected_digit(corrected_angle(raw_angles[i], DIAL_ZERO_OFFSETS[i]))
                    for i in range(n)]
    digits       = list(influenced)

    # dial_influenced_digit detects "A(i+1) just drove A(i)" by looking at A(i+1)'s angle.
    # But dial_influenced_digit for A(i-1) still sees A(i)'s physical angle (near 354°),
    # so it won't fire even though A(i) logically crossed. Propagate the carry manually:
    # if dial i advanced (9→0) and dial i-1 didn't advance on its own, increment i-1.
    for i in range(n - 1, 0, -1):
        if influenced[i] == 0 and raw_corr[i] == 9:
            if not (influenced[i - 1] == 0 and raw_corr[i - 1] == 9):
                digits[i - 1] = (digits[i - 1] + 1) % 10

    fractional = sum(d * 10 ** -(i + 1) for i, d in enumerate(digits))
    return round(integer_part + fractional, 4)


# ── Rollover coverage ──────────────────────────────────────────────────────────

def corrected_fraction(raw_angles: list[float | None]) -> float | None:
    """Fractional reading using the same boundary-corrected logic as assemble_reading."""
    if any(a is None for a in raw_angles):
        return None
    total = sum(
        dial_influenced_digit(i, raw_angles) * 10 ** -(i + 1)
        for i in range(len(raw_angles))
    )
    return round(total, 4)


def corrected_fraction_exit(raw_angles: list[float | None]) -> float | None:
    """Stable fraction for rollover-exit detection; uses corrected_digit only.

    Unlike corrected_fraction, does not apply dial influence — so it only drops
    below ROLLOVER_START when A0 has physically crossed the boundary, not when
    dial_influenced_digit prematurely advances A0 at the 9→0 threshold."""
    if any(a is None for a in raw_angles):
        return None
    total = sum(
        corrected_digit(corrected_angle(raw_angles[i], DIAL_ZERO_OFFSETS[i])) * 10 ** -(i + 1)
        for i in range(len(raw_angles))
    )
    return round(total, 4)


def rollover_coverage(digital: list[int | None],
                      raw_angles: list[float | None],
                      state: dict) -> list[int | None]:
    """
    Override OCR digits during digit drum transitions.

    Rollover in progress:  corrected_fraction >= ROLLOVER_START
    Rollover just complete: fraction wrapped from >= ROLLOVER_START to below it

    D4 always transitions; cascade to D3, D2, ... for each digit that was 9
    in the last accepted reading (e.g. 299→300 triggers D4+D3+D2).

    During transition → force to 9.  After transition → force to 0.
    """
    frac      = corrected_fraction(raw_angles)
    frac_exit = corrected_fraction_exit(raw_angles)
    if frac is None or frac_exit is None:
        return digital

    last = state.get("last_reading")
    if last is None:
        return digital

    last_frac = round(last % 1.0, 4)
    last_digs = [int(c) for c in f"{int(last):05d}"]
    result    = list(digital)

    # pos transitions if ALL less-significant positions (pos+1..4) held 9,
    # causing them to carry and increment this position.
    transitioning = [4]
    for pos in (3, 2, 1, 0):
        if all(last_digs[p] == 9 for p in range(pos + 1, 5)):
            transitioning.append(pos)
        else:
            break

    if frac >= ROLLOVER_START:
        corrected = [pos for pos in transitioning if result[pos] != last_digs[pos]]
        for pos in transitioning:
            result[pos] = last_digs[pos]
        if corrected:
            log.info("rollover: in progress frac=%.4f, corrected digits %s → old", frac, corrected)
    elif last_frac >= ROLLOVER_START and frac_exit < ROLLOVER_START:
        # Use frac_exit (corrected_digit only) to confirm A0 physically crossed.
        # frac alone can drop below ROLLOVER_START prematurely when dial_influenced_digit
        # advances A0 at the 9→0 threshold before A0's angle actually crosses.
        for pos in transitioning:
            result[pos] = (last_digs[pos] + 1) % 10
        log.info("rollover: complete frac=%.4f (was %.4f), forcing digits %s → new",
                 frac, last_frac, transitioning)
    else:
        # No rollover in progress or complete. Integer digits must not change —
        # any difference vs last_digs is an OCR misread (e.g. "1" read as "2").
        reverted = [pos for pos in transitioning if result[pos] != last_digs[pos]]
        if reverted:
            for pos in reverted:
                result[pos] = last_digs[pos]
            log.info("rollover: no transition active, reverted OCR digit error at pos %s", reverted)

    return result


# ── State management ───────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(new_val: float, state: dict) -> tuple[bool, str]:
    last = state.get("last_reading")
    if last is None:
        return True, "first reading"

    last_ts = state.get("last_reading_ts")
    if last_ts is not None:
        elapsed = time.time() - last_ts
        allowed = min(MAX_STEP * max(elapsed, READING_INTERVAL) / READING_INTERVAL,
                      MAX_DELTA_CAP)
    else:
        allowed = MAX_STEP

    delta = new_val - last
    if delta >= MAX_DELTA_CAP:
        return False, (f"jump {delta:+.4f} exceeds cap {MAX_DELTA_CAP:.4f} "
                       f"({last:.4f} → {new_val:.4f})")
    if delta > allowed:
        return False, (f"jump {delta:+.4f} exceeds allowed {allowed:.4f} "
                       f"({last:.4f} → {new_val:.4f})")
    if delta < -JITTER_TOLERANCE:
        return False, (f"decrease {delta:+.4f} exceeds jitter tolerance "
                       f"({last:.4f} → {new_val:.4f})")
    return True, "ok"


# ── Debug annotation ───────────────────────────────────────────────────────────

def annotate(img: np.ndarray, digital: list[int | None],
             raw_angles: list[float | None],
             ocr_digital: list[int | None] | None = None) -> np.ndarray:
    out = img.copy()

    # ── Digital strip ──────────────────────────────────────────────────────────
    for i, (x, y, w, h) in enumerate(DIGITAL_DIGITS):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
        if digital[i] is not None:
            d     = digital[i]
            raw_d = ocr_digital[i] if ocr_digital is not None else d
            if raw_d != d:
                color = (0, 165, 255)   # orange: rollover-corrected
            elif d == 0:
                color = (0, 140, 0)     # dim green: zero (common leading digit)
            else:
                color = (0, 220, 0)     # bright green: significant non-zero
            cv2.putText(out, str(d), (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ── Analog dials ───────────────────────────────────────────────────────────
    for i, (cx, cy, r) in enumerate(ANALOG_DIALS):
        cv2.circle(out, (cx, cy), r, (255, 160, 0), 2)

        # Tick marks at each digit position; 0 = bright yellow, 1-9 = dim gray
        for digit in range(10):
            tick_rad = math.radians((digit * 36 + DIAL_ZERO_OFFSETS[i]) % 360)
            s, ca = math.sin(tick_rad), math.cos(tick_rad)
            tick_len = 9 if digit == 0 else 5
            color    = (0, 220, 255) if digit == 0 else (70, 70, 70)
            thickness = 2 if digit == 0 else 1
            p1 = (int(cx + (r - tick_len) * s), int(cy - (r - tick_len) * ca))
            p2 = (int(cx + r * s),              int(cy - r * ca))
            cv2.line(out, p1, p2, color, thickness)

        # Needle lines: gray = raw angle, red = corrected angle
        raw  = raw_angles[i]
        corr = corrected_angle(raw, DIAL_ZERO_OFFSETS[i]) if raw is not None else None
        for angle, color in ((raw, (120, 120, 120)), (corr, (0, 0, 255))):
            if angle is not None:
                a   = math.radians(angle)
                tip = (int(cx + (r - 12) * math.sin(a)),
                       int(cy - (r - 12) * math.cos(a)))
                cv2.line(out, (cx, cy), tip, color, 2)

        # Centre digit: dial_influenced_digit (what the pipeline uses)
        # Color: cyan = matches corrected_digit; orange = dial influence changed it; dim = zero
        if corr is not None:
            cd    = corrected_digit(corr)
            inf_d = dial_influenced_digit(i, raw_angles)
            if inf_d != cd:
                d_color = (0, 165, 255)  # orange: influence correction applied
            elif inf_d == 0:
                d_color = (0, 180, 180)  # dim cyan: zero
            else:
                d_color = (0, 220, 255)  # bright cyan: non-zero, unmodified
            cv2.putText(out, str(inf_d), (cx - 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, d_color, 2)
        else:
            cv2.putText(out, "?", (cx - 8, cy + 5),
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
            {"unit_of_measurement": "m³", "device_class": "water",
             "state_class": "total_increasing", "friendly_name": "Water Meter"},
        ),
        (
            "sensor.water_meter_liters",
            f"{reading * 1000:.1f}",
            {"unit_of_measurement": "L", "device_class": "water",
             "state_class": "total_increasing", "friendly_name": "Water Meter (L)"},
        ),
    ]
    if flow_lpm is not None:
        entities.append((
            "sensor.water_meter_flow",
            f"{flow_lpm:.3f}",
            {"unit_of_measurement": "L/min", "device_class": "volume_flow_rate",
             "state_class": "measurement", "friendly_name": "Water Meter Flow"},
        ))
    for entity_id, st, attrs in entities:
        try:
            resp = requests.post(
                f"{HA_URL}/api/states/{entity_id}",
                json={"state": st, "attributes": attrs},
                headers=headers,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            log.warning("HA push failed (%s): %s", entity_id, e)


# ── Main ───────────────────────────────────────────────────────────────────────

def process(img: np.ndarray, debug: bool = False,
            state: dict | None = None) -> tuple[float, list, list, list]:
    """Returns (reading, digital, raw_angles, ocr_digital).

    ocr_digital is the raw OCR output before rollover_coverage correction.
    """
    rotated    = rotate_image(img, ROTATE_DEG)
    last_int   = int(state["last_reading"]) if state and "last_reading" in state else None
    digital    = read_digital_digits(rotated, last_int=last_int)
    raw_angles = read_analog_dials(rotated)
    ocr_digital = list(digital)  # snapshot before rollover correction

    if state:
        digital = rollover_coverage(digital, raw_angles, state)

    reading = assemble_reading(digital, raw_angles)

    log.debug("digital=%s  raw_angles=%s  corrected_digits=%s  reading=%.4f",
              digital,
              [f"{a:.1f}" if a is not None else "None" for a in raw_angles],
              [corrected_digit(corrected_angle(a, DIAL_ZERO_OFFSETS[i]))
               if a is not None else "?" for i, a in enumerate(raw_angles)],
              reading)

    if debug:
        ann = annotate(rotated, digital, raw_angles, ocr_digital)
        cv2.imwrite("debug_reading.jpg", ann)
        log.debug("Annotated image saved to debug_reading.jpg")

    return reading, digital, raw_angles, ocr_digital


def _fetch_image(image_path: str | None) -> tuple[np.ndarray, bytes | None]:
    """Returns (img, raw_jpeg). raw_jpeg is None for file-based images."""
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        return img, None
    r = requests.get(CAM_SNAPSHOT_URL, timeout=15)
    r.raise_for_status()
    raw = r.content
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Failed to decode image from camera")
    return img, raw


_snapshot_count = 0


def _save_snapshot(raw: bytes, img: np.ndarray, digital: list[int | None],
                   raw_angles: list[float | None],
                   ocr_digital: list[int | None] | None = None) -> None:
    global _snapshot_count
    if not SNAPSHOT_DIR:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = os.path.join(SNAPSHOT_DIR, "raw")
    ann_dir = os.path.join(SNAPSHOT_DIR, "annotated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f"{ts}.jpg"), "wb") as f:
        f.write(raw)
    rotated = rotate_image(img, ROTATE_DEG)
    cv2.imwrite(os.path.join(ann_dir, f"{ts}.jpg"),
                annotate(rotated, digital, raw_angles, ocr_digital))
    _snapshot_count += 1
    if _snapshot_count % 360 == 0:
        _prune_snapshots()


def _prune_snapshots() -> None:
    if not SNAPSHOT_DIR or SNAPSHOT_MAX_AGE_DAYS is None:
        return
    cutoff = time.time() - SNAPSHOT_MAX_AGE_DAYS * 86400
    pruned = 0
    for subdir in ("raw", "annotated"):
        path = os.path.join(SNAPSHOT_DIR, subdir)
        try:
            for entry in os.scandir(path):
                if entry.is_file() and entry.name.endswith(".jpg"):
                    if entry.stat().st_mtime < cutoff:
                        os.unlink(entry.path)
                        pruned += 1
        except OSError:
            pass
    if pruned:
        log.info("pruned %d snapshots older than %.1f days", pruned, SNAPSHOT_MAX_AGE_DAYS)


def _run_once(image_path: str | None, debug: bool, no_guard: bool,
              last_reading: float | None = None, push: bool = False) -> float | None:
    """Fetch, process, validate, and optionally push. Returns accepted reading or None."""
    state = load_state()
    if last_reading is not None:
        state["last_reading"] = last_reading
        state.pop("last_reading_ts", None)
    try:
        img, raw_jpeg = _fetch_image(image_path)
        reading, digital, raw_angles, ocr_digital = process(img, debug=debug, state=state)
        if raw_jpeg is not None:
            _save_snapshot(raw_jpeg, img, digital, raw_angles, ocr_digital)
    except (ValueError, RuntimeError, requests.exceptions.RequestException) as e:
        log.error("%s", e)
        return None

    if no_guard:
        print(f"{reading:.4f}")
        if push:
            push_to_ha(reading)
        return reading

    ok, reason = validate(reading, state)
    if not ok:
        log.warning("Rejected: %s", reason)
        return None

    now      = time.time()
    last_ts  = state.get("last_reading_ts")
    last_val = state.get("last_reading")
    flow_lpm = None
    if last_val is not None and last_ts is not None:
        elapsed = now - last_ts
        if elapsed <= 0:
            log.warning("flow rate skipped: elapsed=%.3fs (clock went backwards?)", elapsed)
        elif FLOW_MAX_AGE is not None and elapsed > FLOW_MAX_AGE:
            log.debug("flow rate skipped: elapsed=%.1fs exceeds FLOW_MAX_AGE=%.1fs",
                      elapsed, FLOW_MAX_AGE)
        else:
            flow_lpm = max(0.0, (reading - last_val) * 1000 / elapsed * 60)

    state["last_reading"]    = reading
    state["last_reading_ts"] = now
    save_state(state)
    flow_str = f"{flow_lpm:.3f} L/min" if flow_lpm is not None else "n/a"
    log.info("accepted reading=%.4f  flow=%s", reading, flow_str)
    if push:
        push_to_ha(reading, flow_lpm)
    return reading


def _reannotate_all() -> None:
    """Re-process all existing raw snapshots and overwrite their annotated images."""
    if not SNAPSHOT_DIR:
        log.error("SNAPSHOT_DIR not configured — cannot reannotate")
        return
    raw_dir = os.path.join(SNAPSHOT_DIR, "raw")
    ann_dir = os.path.join(SNAPSHOT_DIR, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(raw_dir, "*.jpg")))
    if not paths:
        log.info("No raw snapshots found in %s", raw_dir)
        return
    log.info("Reannotating %d snapshots …", len(paths))
    ok = err = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            log.warning("Cannot read %s — skipped", path)
            err += 1
            continue
        rotated    = rotate_image(img, ROTATE_DEG)
        digital    = read_digital_digits(rotated)
        raw_angles = read_analog_dials(rotated)
        # No state/rollover for reannotation — show raw OCR as-is
        out_path = os.path.join(ann_dir, os.path.basename(path))
        cv2.imwrite(out_path, annotate(rotated, digital, raw_angles))
        ok += 1
    log.info("Reannotated %d images (%d errors)", ok, err)


def main() -> None:
    ap = argparse.ArgumentParser(description="Read water meter from camera.")
    ap.add_argument("--image",    help="Load from file instead of live camera")
    ap.add_argument("--debug",    action="store_true", help="Write debug_reading.jpg")
    ap.add_argument("--no-guard", action="store_true", help="Skip sanity guards")
    ap.add_argument("--loop",         action="store_true",
                    help="Run continuously on READING_INTERVAL")
    ap.add_argument("--interval",     type=float, default=None,
                    help="Override READING_INTERVAL (seconds)")
    ap.add_argument("--last-reading", type=float, default=None,
                    help="Override last_reading baseline for this run (not allowed with --loop)")
    ap.add_argument("--push",         action="store_true",
                    help="Push result to HA even on a one-off run")
    ap.add_argument("--reannotate",   action="store_true",
                    help="Re-annotate all existing raw snapshots and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.reannotate:
        _reannotate_all()
        sys.exit(0)

    if args.last_reading is not None and args.loop:
        ap.error("--last-reading cannot be used with --loop")

    push = args.loop or args.push

    if not args.loop:
        result = _run_once(args.image, args.debug, args.no_guard, args.last_reading, push=push)
        sys.exit(0 if result is not None else 1)

    interval = args.interval if args.interval is not None else READING_INTERVAL
    log.info("Loop mode: interval=%.1fs  cam=%s  ha=%s",
             interval, CAM_SNAPSHOT_URL, HA_URL or "(not configured)")

    while True:
        t0 = time.monotonic()
        _run_once(args.image, args.debug, args.no_guard, push=True)
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, interval - elapsed))


if __name__ == "__main__":
    main()
