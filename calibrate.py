#!/usr/bin/env python3
"""Calibration helper for meter_reader.py.

Usage:
  python3 calibrate.py --image snapshots/.../foo.jpg --scan-rotation
      Scan ROTATE_DEG in 0.5° steps and print counter strip slope.

  python3 calibrate.py --image snapshots/.../foo.jpg --rotation 68.0 --grid
      Save grid.jpg showing digit crops and dial circles at the given rotation.

  python3 calibrate.py --image snapshots/.../foo.jpg --rotation 68.0 --hough
      Run Hough circle detection and print candidate dial centers.

  python3 calibrate.py --image snapshots/.../foo.jpg --rotation 68.0 --interactive
      Click-based calibration: define strip corners and dial positions visually.
      Prints ready-to-paste constants for meter_reader.py.
      Controls: left-click to place, U = undo, Enter = confirm, Q/Esc = quit.
"""

import argparse
import math
import sys

import cv2
import numpy as np

# ── Import constants from meter_reader ─────────────────────────────────────────
sys.path.insert(0, ".")
import meter_reader as mr


def rotate_image(img, deg):
    return mr.rotate_image(img, deg)


# ── Rotation scan ──────────────────────────────────────────────────────────────
def strip_slope(rotated: np.ndarray, sx: int, sy: int, sw: int, sh: int) -> float:
    """Return brightness-centroid slope across the strip (px/px).

    A horizontal counter has slope ≈ 0. We split the strip into left and right
    halves and measure the vertical centroid shift.
    """
    strip = rotated[sy:sy + sh, sx:sx + sw]
    gray  = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(float)
    # invert so dark digits are bright → high signal where digits are
    inv   = 255.0 - gray

    h, w = inv.shape
    left  = inv[:, :w // 2]
    right = inv[:, w // 2:]

    def cy(region):
        col_sum = region.sum(axis=1)
        if col_sum.sum() == 0:
            return region.shape[0] / 2
        ys = np.arange(region.shape[0])
        return float((ys * col_sum).sum() / col_sum.sum())

    return (cy(right) - cy(left)) / (w / 2)


def scan_rotation(img: np.ndarray, deg_start: float, deg_end: float, deg_step: float,
                  strip: tuple = None):
    sx, sy, sw, sh = strip if strip else mr._DIGITAL_STRIP
    best_deg, best_slope = None, float("inf")
    print(f"{'deg':>7}  {'slope':>10}")
    for i in range(round((deg_end - deg_start) / deg_step) + 1):
        deg = deg_start + i * deg_step
        rot = rotate_image(img, deg)
        s   = strip_slope(rot, sx, sy, sw, sh)
        marker = " ←" if abs(s) < abs(best_slope) else ""
        print(f"{deg:7.1f}  {s:10.5f}{marker}")
        if abs(s) < abs(best_slope):
            best_deg, best_slope = deg, s
    print(f"\nBest rotation: {best_deg}° (slope={best_slope:.5f})")


# ── Grid view ─────────────────────────────────────────────────────────────────
def save_grid(img: np.ndarray, deg: float, out_path: str = "grid.jpg"):
    rotated = rotate_image(img, deg)
    out     = rotated.copy()

    # Draw digital digit crops
    for i, (x, y, w, h) in enumerate(mr.DIGITAL_DIGITS):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(out, f"D{i}", (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

    # Draw strip
    sx, sy, sw, sh = mr._DIGITAL_STRIP
    cv2.rectangle(out, (sx, sy), (sx + sw, sy + sh), (0, 200, 200), 2)

    # Draw dial circles
    for i, (cx, cy, r, _, _) in enumerate(mr.ANALOG_DIALS):
        cv2.circle(out, (cx, cy), r, (255, 160, 0), 2)
        cv2.putText(out, f"A{i}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 160, 0), 2)

    cv2.imwrite(out_path, out)
    print(f"Grid saved to {out_path}  (rotated {deg}°, size {rotated.shape[1]}×{rotated.shape[0]})")


# ── Hough circle detection ────────────────────────────────────────────────────
def detect_hough(img: np.ndarray, deg: float):
    rotated = rotate_image(img, deg)
    gray    = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=80,
        param1=100, param2=30,
        minRadius=50, maxRadius=90,
    )

    if circles is None:
        print("No circles detected.")
        return

    circles = np.round(circles[0]).astype(int)
    circles = sorted(circles, key=lambda c: c[0])  # sort left→right by cx
    print(f"Detected {len(circles)} circle(s) at rotation {deg}°:")
    for c in circles:
        print(f"  cx={c[0]:4d}  cy={c[1]:4d}  r={c[2]:3d}")

    # Save annotated image
    out = rotated.copy()
    for c in circles:
        cv2.circle(out, (c[0], c[1]), c[2], (0, 200, 255), 2)
        cv2.circle(out, (c[0], c[1]), 3, (0, 200, 255), -1)
    cv2.imwrite("hough.jpg", out)
    print("Hough overlay saved to hough.jpg")


# ── Interactive calibration ───────────────────────────────────────────────────
def interactive_calibration(img: np.ndarray, deg: float):
    rotated = rotate_image(img, deg)
    h, w = rotated.shape[:2]

    MAX_DIM = 1100
    scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
    dw, dh = int(w * scale), int(h * scale)

    DIAL_LABELS = ["×0.1 m³", "×0.01 m³", "×0.001 m³", "×0.0001 m³"]
    INSTRUCTIONS = [
        "1/10  Click TOP-LEFT corner of digit strip",
        "2/10  Click BOTTOM-RIGHT corner of digit strip",
        "3/10  Click CENTER of dial 1 (×0.1 m³)",
        "4/10  Click RIM of dial 1 (×0.1 m³)",
        "5/10  Click CENTER of dial 2 (×0.01 m³)",
        "6/10  Click RIM of dial 2 (×0.01 m³)",
        "7/10  Click CENTER of dial 3 (×0.001 m³)",
        "8/10  Click RIM of dial 3 (×0.001 m³)",
        "9/10  Click CENTER of dial 4 (×0.0001 m³)",
        "10/10 Click RIM of dial 4 (×0.0001 m³)",
    ]

    clicks = []  # (x, y) in full-image coordinates

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 10:
            clicks.append((int(x / scale), int(y / scale)))

    def draw():
        out = rotated.copy()
        n = len(clicks)

        if n >= 2:
            cv2.rectangle(out, clicks[0], clicks[1], (0, 220, 0), 2)
            cv2.putText(out, "strip", (clicks[0][0], clicks[0][1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
        elif n == 1:
            cv2.circle(out, clicks[0], 6, (0, 220, 0), -1)

        for i in range(4):
            ci, ri = 2 + i * 2, 3 + i * 2
            if n > ri:
                cx, cy = clicks[ci]
                r = int(math.hypot(clicks[ri][0] - cx, clicks[ri][1] - cy))
                cv2.circle(out, (cx, cy), r, (255, 160, 0), 2)
                cv2.circle(out, (cx, cy), 4, (255, 160, 0), -1)
                cv2.putText(out, DIAL_LABELS[i], (cx - 10, cy - r - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 160, 0), 2)
            elif n > ci:
                cv2.circle(out, clicks[ci], 6, (255, 160, 0), -1)

        disp = cv2.resize(out, (dw, dh))

        if n < 10:
            bar_text = INSTRUCTIONS[n] + "   [U=undo  Q=quit]"
        else:
            bar_text = "All done — press Enter to confirm, U to undo, Q to quit"
        cv2.rectangle(disp, (0, 0), (dw, 38), (0, 0, 0), -1)
        cv2.putText(disp, bar_text, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        return disp

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibration", on_click)
    print("Interactive calibration — follow the on-screen instructions.")
    print("  Left-click to place  |  U = undo  |  Enter = confirm  |  Q/Esc = quit")

    while True:
        cv2.imshow("Calibration", draw())
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), 27):
            cv2.destroyAllWindows()
            print("Cancelled.")
            return
        if key == ord('u') and clicks:
            clicks.pop()
        if key in (13, 10) and len(clicks) == 10:
            break

    cv2.destroyAllWindows()

    tl, br = clicks[0], clicks[1]
    sx = min(tl[0], br[0])
    sy = min(tl[1], br[1])
    sw = abs(br[0] - tl[0])
    sh = abs(br[1] - tl[1])

    digit_w = sw // 5
    digits  = [(sx + i * digit_w, sy, digit_w, sh) for i in range(5)]

    strip_cx = sx + sw // 2
    strip_cy = sy + sh // 2

    offsets = []
    for i in range(4):
        cx, cy = clicks[2 + i * 2]
        r = int(math.hypot(clicks[3 + i * 2][0] - cx, clicks[3 + i * 2][1] - cy))
        offsets.append((cx - strip_cx, cy - strip_cy, r))

    print(f"\n# ── Paste into meter_reader.py {'─' * 38}")
    print(f"ROTATE_DEG = {deg}")
    print(f"_DIGITAL_STRIP = ({sx}, {sy}, {sw}, {sh})")
    print("DIGITAL_DIGITS = [")
    for x, y, dw2, dh2 in digits:
        print(f"    ({x}, {y}, {dw2}, {dh2}),")
    print("]  # auto-split; adjust x/w per digit if OCR struggles")
    print("\n_ANALOG_DIAL_OFFSETS = [  # (dx, dy, r) from strip centre")
    for i, (dx, dy, r) in enumerate(offsets):
        print(f"    ({dx:+d}, {dy:+d}, {r}, False, False),  # {DIAL_LABELS[i]}")
    print("]")
    print(f"# {'─' * 55}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Calibration helper for meter_reader.py")
    ap.add_argument("--image", required=True, help="Input snapshot path")
    ap.add_argument("--rotation", type=float, default=None,
                    help="Rotation angle to use (overrides meter_reader.ROTATE_DEG)")
    ap.add_argument("--scan-rotation", action="store_true",
                    help="Scan rotation in 0.5° steps around current ROTATE_DEG ± 5°")
    ap.add_argument("--grid", action="store_true",
                    help="Save grid.jpg with digit/dial overlays")
    ap.add_argument("--hough", action="store_true",
                    help="Run Hough circle detection")
    ap.add_argument("--interactive", action="store_true",
                    help="Click-based calibration: define strip and dial positions visually")
    ap.add_argument("--strip", type=int, nargs=4, metavar=("X","Y","W","H"),
                    help="Strip ROI to use for rotation scan (overrides meter_reader._DIGITAL_STRIP)")
    ap.add_argument("--scan-start", type=float, default=None)
    ap.add_argument("--scan-end",   type=float, default=None)
    ap.add_argument("--scan-step",  type=float, default=0.5)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Cannot read: {args.image}")

    deg = args.rotation if args.rotation is not None else mr.ROTATE_DEG

    if args.scan_rotation:
        start = args.scan_start if args.scan_start is not None else mr.ROTATE_DEG - 5.0
        end   = args.scan_end   if args.scan_end   is not None else mr.ROTATE_DEG + 5.0
        strip = tuple(args.strip) if args.strip else None
        scan_rotation(img, start, end, args.scan_step, strip=strip)

    if args.grid:
        save_grid(img, deg)

    if args.hough:
        detect_hough(img, deg)

    if args.interactive:
        interactive_calibration(img, deg)


if __name__ == "__main__":
    main()
