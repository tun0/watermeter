#!/usr/bin/env python3
"""
meter_reader.py — standalone water meter reader, replacing nohn/watermeter.

Pipeline:
  1. Fetch /raw from overlay-proxy (undistorted, unrotated)
  2. Rotate 184°
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
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
PROXY_RAW_URL = "http://localhost:8081/raw"
ROTATE_DEG    = 184

# Digital digit crop boxes in rotated-image coordinates (from config.php)
# Each entry: (x, y, w, h) — ordered most-significant to least-significant
DIGITAL_DIGITS = [
    (252, 124, 41, 55),   # 10,000 m³ digit (normally 0)
    (310, 126, 40, 55),
    (368, 127, 35, 53),
    (423, 124, 33, 55),
    (480, 124, 34, 54),
]

# Full strip covering all digital digits; used by the strip OCR path.
_DIGITAL_STRIP = (252, 124, 263, 57)   # x, y, w, h  (252→515)

# Analog dial definitions: (cx, cy, radius, dark_needle, flip)
# Ordered most-significant → least-significant
ANALOG_DIALS = [
    (671, 288, 69, False, False),
    (606, 446, 73, False, False),
    (449, 507, 66, True,  False),
    (287, 438, 68, True,  False),
]

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

MAX_STEP       = 2.0    # reject reading jumps larger than this (m³)
ALLOW_DECREASE = False

STATE_FILE = Path(__file__).parent / ".meter_state.json"

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
    best_conf, best_digit = -1, None

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
            if text.isdigit() and int(conf) > best_conf:
                best_conf, best_digit = int(conf), int(text)

    return best_digit


def read_digital_digits(img: np.ndarray) -> list[int | None]:
    """Return list of 5 ints (or None per failed digit) from the digital counter.

    Primary path: reads the full digit strip as one unit with PSM 8 — more
    reliable than per-digit PSM 10 for this meter's font.  Falls back to
    per-digit OCR if the strip result doesn't yield exactly 5 digits.
    """
    sx, sy, sw, sh = _DIGITAL_STRIP
    strip = img[sy:sy + sh, sx:sx + sw]
    strip3x = cv2.resize(strip, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(cv2.cvtColor(strip3x, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.copyMakeBorder(thresh, 12, 12, 12, 12,
                                cv2.BORDER_CONSTANT, value=255)
    for psm in (8, 7, 6):
        cfg = f"--psm {psm} --oem 3 -c tessedit_char_whitelist=0123456789"
        result = pytesseract.image_to_string(thresh, config=cfg).strip()
        if len(result) == len(DIGITAL_DIGITS) and result.isdigit():
            return [int(c) for c in result]

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
    return round(integer_part + fractional, 4)


# ── Sanity guards ──────────────────────────────────────────────────────────────
def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
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


# ── Main ───────────────────────────────────────────────────────────────────────
def process(img: np.ndarray, debug: bool = False) -> float:
    rotated      = rotate_image(img, ROTATE_DEG)
    digital      = read_digital_digits(rotated)
    angles_raw   = read_analog_dials(rotated)
    angles_cor   = correct_gear_lash(angles_raw)
    reading      = assemble_reading(digital, angles_cor)

    log.info("digital=%s  raw_angles=%s  corrected_digits=%s  reading=%.4f",
             digital,
             [f"{a:.1f}" if a is not None else "None" for a in angles_raw],
             [angle_to_digit(a) if a is not None else "?" for a in angles_cor],
             reading)

    if debug:
        ann = annotate(rotated, digital, angles_raw, angles_cor)
        cv2.imwrite("debug_reading.jpg", ann)
        log.info("Annotated image saved to debug_reading.jpg")

    return reading


def main() -> None:
    ap = argparse.ArgumentParser(description="Read water meter from overlay-proxy snapshot.")
    ap.add_argument("--image",    help="Load from file instead of live proxy")
    ap.add_argument("--debug",    action="store_true", help="Write debug_reading.jpg")
    ap.add_argument("--no-guard", action="store_true", help="Skip sanity guards")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Cannot read image: {args.image}")
    else:
        r = requests.get(PROXY_RAW_URL, timeout=15)
        r.raise_for_status()
        arr = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit("Failed to decode image from proxy")

    try:
        reading = process(img, debug=args.debug)
    except ValueError as e:
        log.error("%s", e)
        sys.exit(1)

    if not args.no_guard:
        state = load_state()
        ok, reason = validate(reading, state)
        if not ok:
            log.warning("Rejected: %s", reason)
            print(f"REJECTED {reading:.4f}")
            sys.exit(2)
        state["last_reading"] = reading
        save_state(state)

    print(f"{reading:.4f}")


if __name__ == "__main__":
    main()
