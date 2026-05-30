#!/usr/bin/env python3
"""
Overlay crisp digit rings onto water meter sub-dials.

Fetches a fresh snapshot, draws clean 0-9 number rings around each sub-dial,
and writes the composited image so OCR tools (e.g. watermeter) can read them.

Usage:
    python3 meter_overlay.py [--verify] [--output /tmp/meter_overlay.jpg]

    --verify  : draws circles + axes only (no digit overlay) to check calibration
"""

import argparse
import math
import sys
import time

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ── Camera ──────────────────────────────────────────────────────────────────
STREAM_URL   = "http://192.168.x.x:8080/"   # MJPEG keepalive
SNAPSHOT_URL = "http://192.168.x.x/"         # single JPEG
LIGHT_URL    = "http://192.168.x.x:81/light/esp-cam-2_light"
TIMEOUT      = 10

# ── Dial geometry (cx, cy, radius, angle_offset_deg) ────────────────────────
# angle_offset_deg: where digit "0" sits, measured clockwise from 12 o'clock.
#   0   → 12 o'clock (top)
#   90  → 3 o'clock (right)
#   -90 → 9 o'clock (left)
# Run with --verify to see circles drawn on the raw image.
DIALS = [
    ( 990, 265, 120, -64),  # dial 0 — top-right
    (1185, 490, 120, -64),  # dial 1 — right-upper
    (1165, 795, 120, -64),  # dial 2 — right-lower
    ( 915, 995, 120, -64),  # dial 3 — bottom
]

# Clockwise digit order 0→9
DIGITS = list("0123456789")

# ── Overlay style ────────────────────────────────────────────────────────────
FONT_SIZE      = 48        # pt
DIGIT_MARGIN   = 14        # px inward from circle edge where digit centres sit
DIGIT_COLOR    = (0, 0, 0, 255)        # solid black
DIGIT_SHADOW   = (255, 255, 255, 255)  # white outline for contrast
CIRCLE_COLOR   = (0, 255, 0, 160)     # verify mode circle
CIRCLE_WIDTH   = 2

# Rotate each digit glyph this many degrees CCW before placing it.
# Set to match the corrective rotation the watermeter app applies to the image,
# so digits appear upright after that rotation.  Try 90 or -90 if unsure.
DIGIT_ROTATION = 65  # degrees CCW — matches sourceImageRotate in watermeter config.php


def warm_camera():
    """Hit the MJPEG stream briefly to wake the camera from idle."""
    try:
        r = requests.get(STREAM_URL, stream=True, timeout=3)
        r.close()
    except Exception:
        pass
    time.sleep(0.5)


def fetch_snapshot() -> np.ndarray:
    warm_camera()
    for attempt in range(3):
        try:
            r = requests.get(SNAPSHOT_URL, timeout=TIMEOUT)
            r.raise_for_status()
            arr = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception as e:
            print(f"  snapshot attempt {attempt+1} failed: {e}", file=sys.stderr)
            time.sleep(1)
    raise RuntimeError("Could not fetch snapshot after 3 attempts")


def load_local(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def draw_verify(img: np.ndarray) -> np.ndarray:
    """Draw circles and cross-hairs so calibration can be checked."""
    out = img.copy()
    for i, (cx, cy, r, ang) in enumerate(DIALS):
        cv2.circle(out, (cx, cy), r, (0, 255, 0), CIRCLE_WIDTH)
        cv2.circle(out, (cx, cy), 4, (0, 100, 255), -1)
        cv2.line(out, (cx - r - 10, cy), (cx + r + 10, cy), (0, 200, 255), 1)
        cv2.line(out, (cx, cy - r - 10), (cx, cy + r + 10), (0, 200, 255), 1)
        cv2.putText(out, f"dial {i} ang={ang}", (cx - 30, cy - r - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def get_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size)
    except OSError:
        pass
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_digit_ring(overlay: Image.Image, cx: int, cy: int, r: int, font, angle_offset: float = 0):
    """Paste rotated digit glyphs evenly spaced around the dial circle."""
    n = len(DIGITS)
    pad = 4
    for i, digit in enumerate(DIGITS):
        angle_deg = angle_offset + i * (360 / n)
        angle_rad = math.radians(angle_deg)
        dx = cx + (r - DIGIT_MARGIN) * math.sin(angle_rad)
        dy = cy - (r - DIGIT_MARGIN) * math.cos(angle_rad)

        # Render digit onto a small transparent cell
        bbox = font.getbbox(digit)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cell = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(cell)
        d.text((pad + 1, pad + 1), digit, font=font, fill=DIGIT_SHADOW)
        d.text((pad, pad), digit, font=font, fill=DIGIT_COLOR)

        # Rotate the glyph (expand=True keeps the full rotated image)
        if DIGIT_ROTATION != 0:
            cell = cell.rotate(DIGIT_ROTATION, expand=True)

        # Paste centred at (dx, dy)
        cw, ch = cell.size
        overlay.paste(cell, (int(dx - cw / 2), int(dy - ch / 2)), cell)


def apply_overlay(img_bgr: np.ndarray) -> np.ndarray:
    """Composite a digit ring overlay onto the meter image."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    base = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))

    font = get_font(FONT_SIZE)

    for cx, cy, r, angle_offset in DIALS:
        draw_digit_ring(overlay, cx, cy, r, font, angle_offset)

    composited = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.array(composited), cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="Draw calibration circles instead of digit overlay")
    parser.add_argument("--input", default=None,
                        help="Use local image instead of fetching snapshot")
    parser.add_argument("--output", default="/tmp/meter_overlay.jpg",
                        help="Output path (default: /tmp/meter_overlay.jpg)")
    args = parser.parse_args()

    if args.input:
        img = load_local(args.input)
        print(f"Using local image {args.input} ({img.shape[1]}x{img.shape[0]})")
    else:
        print("Fetching snapshot from camera…")
        img = fetch_snapshot()
        print(f"Snapshot OK ({img.shape[1]}x{img.shape[0]})")

    if args.verify:
        out = draw_verify(img)
        msg = "verify"
    else:
        out = apply_overlay(img)
        msg = "overlay"

    cv2.imwrite(args.output, out)
    print(f"Saved {msg} → {args.output}")


if __name__ == "__main__":
    main()
