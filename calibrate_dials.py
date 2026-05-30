#!/usr/bin/env python3
"""
Dial calibration helper for watermeter analog gauges.

Fetches the proxy image, applies watermeter's rotation, detects the four
analog dial circles via HoughCircles, and outputs ready-to-paste config.php
bounding box entries.

Usage:
    python3 calibrate_dials.py [--input /path/to/image.jpg] [--rotate DEG]
    python3 calibrate_dials.py --output /tmp/debug_circles.jpg
"""

import argparse
import sys
import cv2
import numpy as np
import requests

PROXY_URL  = "http://localhost:8081/raw"
ROTATE_DEG = 184   # must match sourceImageRotate in config.php

# Lens correction — keep in sync with overlay-proxy/app.py
LENS_K1      = -0.20
LENS_K2      =  0.05
LENS_F_SCALE =  0.82


def fetch_image(path: str | None) -> np.ndarray:
    if path:
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"Cannot read {path}")
        return img
    r = requests.get(PROXY_URL, timeout=15)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit("Failed to decode camera image")
    return img


def undistort(img: np.ndarray) -> np.ndarray:
    if LENS_K1 == 0.0:
        return img
    h, w = img.shape[:2]
    f = w * LENS_F_SCALE
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1   ]], dtype=np.float64)
    D = np.array([LENS_K1, LENS_K2, 0.0, 0.0], dtype=np.float64)
    return cv2.undistort(img, K, D)


def rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate keeping full canvas (matches Imagick rotateImage behaviour)."""
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


def find_needle_pivot(img_bgr: np.ndarray, cx: int, cy: int, r: int,
                      margin: int = 15) -> tuple[int, int] | None:
    """
    Within the dial region, find the red needle and return its pivot (tail) point.
    The teardrop needle has a narrow tail at the pivot and a round head at the tip;
    we find the end of the fitted bounding rectangle that has fewer red pixels.
    Returns (px, py) in full-image coordinates, or None if no red blob found.
    """
    x0 = max(0, cx - r - margin)
    y0 = max(0, cy - r - margin)
    x1 = min(img_bgr.shape[1], cx + r + margin)
    y1 = min(img_bgr.shape[0], cy + r + margin)
    region = img_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Red wraps around 0/180 in HSV — permissive thresholds for dark/maroon needles
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   50, 30]), np.array([20, 255, 255])),
        cv2.inRange(hsv, np.array([155, 50, 30]), np.array([180, 255, 255])),
    )
    # Clean up noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 80:
        return None

    rect = cv2.minAreaRect(cnt)
    box_center, (bw, bh), angle = rect
    # Ensure long axis is bw
    if bh > bw:
        bw, bh = bh, bw
        angle += 90
    angle_rad = np.radians(angle)
    direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    half_len = bw / 2
    end_a = np.array(box_center) + direction * half_len
    end_b = np.array(box_center) - direction * half_len

    def red_pixel_count(pt: np.ndarray) -> int:
        probe_r = max(4, int(bh * 0.35))
        px, py = int(pt[0]), int(pt[1])
        patch = mask[max(0, py - probe_r):py + probe_r,
                     max(0, px - probe_r):px + probe_r]
        return int(patch.sum()) // 255

    # Narrow end (fewer red pixels) = tail = pivot
    tail = end_a if red_pixel_count(end_a) < red_pixel_count(end_b) else end_b
    return (int(tail[0]) + x0, int(tail[1]) + y0)


def detect_dials(img: np.ndarray, n_dials: int = 4,
                 min_r: int = 50, max_r: int = 100) -> list[tuple[int, int, int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE to boost local contrast of dial graduation marks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)

    circles = None
    for dp in (1, 1.2, 1.5):
        for p1 in (60, 50, 40):
            for p2 in (30, 25, 20):
                c = cv2.HoughCircles(
                    gray, cv2.HOUGH_GRADIENT, dp=dp,
                    minDist=min_r * 2,
                    param1=p1, param2=p2,
                    minRadius=min_r, maxRadius=max_r,
                )
                if c is not None and len(c[0]) >= n_dials:
                    circles = c
                    break
            if circles is not None:
                break
        if circles is not None:
            break

    if circles is None:
        print("WARNING: fewer than expected circles detected", file=sys.stderr)
        return []

    detected = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    # Keep the n_dials circles with tightest radius spread (most dial-like cluster)
    detected.sort(key=lambda c: c[2])
    return detected[:n_dials]


def show_grid_and_blobs(img: np.ndarray, output: str) -> None:
    """
    Overlay a 50px coordinate grid and mark all red blobs with their centroids.
    Save to output. Use this to visually identify dial centers and read off coords.
    """
    out = img.copy()
    h, w = out.shape[:2]

    # Coordinate grid
    for x in range(0, w, 50):
        cv2.line(out, (x, 0), (x, h), (60, 60, 60), 1)
        if x % 100 == 0:
            cv2.putText(out, str(x), (x + 2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)
    for y in range(0, h, 50):
        cv2.line(out, (0, y), (w, y), (60, 60, 60), 1)
        if y % 100 == 0:
            cv2.putText(out, str(y), (2, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

    # All red blobs in the full image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   50, 30]), np.array([20, 255, 255])),
        cv2.inRange(hsv, np.array([155, 50, 30]), np.array([180, 255, 255])),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Fit bounding rect → needle direction + tail end
        rect = cv2.minAreaRect(cnt)
        box_center, (bw, bh), angle = rect
        if bh > bw:
            bw, bh = bh, bw
            angle += 90
        angle_rad = np.radians(angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        half_len = bw / 2
        end_a = np.array(box_center) + direction * half_len
        end_b = np.array(box_center) - direction * half_len

        def red_count(pt):
            r = max(4, int(bh * 0.35))
            px, py = int(pt[0]), int(pt[1])
            patch = mask[max(0, py-r):py+r, max(0, px-r):px+r]
            return int(patch.sum()) // 255

        tail = end_a if red_count(end_a) < red_count(end_b) else end_b
        tx, ty = int(tail[0]), int(tail[1])

        cv2.drawContours(out, [cnt], -1, (0, 100, 255), 1)
        cv2.circle(out, (cx, cy), 4, (0, 180, 255), -1)   # centroid (orange)
        cv2.circle(out, (tx, ty), 6, (0, 255, 0),   2)    # tail/pivot (green ring)
        label = f"({tx},{ty})"
        cv2.putText(out, label, (tx + 6, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imwrite(output, out)
    print(f"Grid + blobs image → {output}")
    print("Green rings = estimated needle pivot (tail).  Orange dots = blob centroid.")
    print("Read the (x,y) coordinates and use --override N:CX,CY,R to set each dial center.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="Local image file instead of fetching from proxy")
    ap.add_argument("--rotate", type=float, default=ROTATE_DEG,
                    help=f"Rotation degrees (default: {ROTATE_DEG})")
    ap.add_argument("--output", default="/tmp/dial_circles.jpg",
                    help="Annotated debug image output path")
    ap.add_argument("--margin", type=int, default=10,
                    help="Extra pixels around detected radius for bounding box")
    ap.add_argument("--min-radius", type=int, default=50,
                    help="Minimum circle radius to detect (default: 50)")
    ap.add_argument("--max-radius", type=int, default=100,
                    help="Maximum circle radius to detect (default: 100)")
    ap.add_argument("--grid", action="store_true",
                    help="Output rotated image with coordinate grid and red blob pivots "
                         "for manual center identification (skips Hough detection)")
    ap.add_argument("--no-refine", action="store_true",
                    help="Skip red-needle pivot refinement; use Hough/override centers as-is")
    ap.add_argument("--override", action="append", default=[], metavar="N:CX,CY,R",
                    help="Override Hough result for detected dial N (1-based) with manual "
                         "center and radius, e.g. --override 2:605,421,65. "
                         "Use the annotated image to identify the correct center.")
    args = ap.parse_args()

    # Parse overrides: {1-based index → (cx, cy, r)}
    overrides: dict[int, tuple[int, int, int]] = {}
    for ov in args.override:
        try:
            idx_s, coords = ov.split(":", 1)
            parts = [int(v) for v in coords.split(",")]
            overrides[int(idx_s)] = (parts[0], parts[1], parts[2])
        except Exception:
            sys.exit(f"Bad --override format '{ov}', expected N:CX,CY,R")

    print("Fetching image…")
    img = fetch_image(args.input)
    print(f"  raw size: {img.shape[1]}×{img.shape[0]}")

    if args.input:
        img = undistort(img)
        print(f"  undistorted (k1={LENS_K1})")

    print(f"Rotating {args.rotate}°…")
    rotated = rotate_image(img, args.rotate)
    print(f"  rotated size: {rotated.shape[1]}×{rotated.shape[0]}")

    if args.grid:
        show_grid_and_blobs(rotated, args.output)
        return

    print("Detecting dials via HoughCircles…")
    dials = detect_dials(rotated, min_r=args.min_radius, max_r=args.max_radius)
    if not dials:
        sys.exit("No dials detected — check lighting or try --input with a cleaner image")

    # Apply manual overrides (extend list if needed)
    dials = list(dials)
    for idx, val in overrides.items():
        if idx >= 1:
            while len(dials) < idx:
                dials.append((0, 0, 65))
            old = dials[idx - 1]
            dials[idx - 1] = val
            print(f"  override dial {idx}: {old} → {val}")

    print(f"  using {len(dials)} circles ({len(overrides)} overridden)")

    # Refine each Hough center using the red needle's pivot (tail)
    refined = []
    for i, (cx, cy, r) in enumerate(dials):
        if args.no_refine:
            refined.append((cx, cy, r))
        else:
            pivot = find_needle_pivot(rotated, cx, cy, r, margin=args.margin)
            if pivot:
                print(f"  dial {i+1}: Hough ({cx},{cy}) → needle pivot ({pivot[0]},{pivot[1]})")
                refined.append((pivot[0], pivot[1], r))
            else:
                print(f"  dial {i+1}: no red needle found, keeping Hough center ({cx},{cy})")
                refined.append((cx, cy, r))

    annotated = rotated.copy()
    colours = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (200, 0, 255)]
    for i, ((hx, hy, _), (cx, cy, r)) in enumerate(zip(dials, refined)):
        col = colours[i % len(colours)]
        cv2.circle(annotated, (cx, cy), r, col, 2)          # refined circle
        cv2.circle(annotated, (cx, cy), 6, col, -1)          # refined center
        cv2.circle(annotated, (hx, hy), 4, (128, 128, 128), -1)  # original Hough (grey)
        cv2.putText(annotated, str(i + 1), (cx - 8, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
    cv2.imwrite(args.output, annotated)
    print(f"  annotated image → {args.output}")

    print()
    print("─" * 62)
    print("Suggested analogGauges for config.php (check order against meter):")
    print("─" * 62)
    for i, (cx, cy, r) in enumerate(refined):
        m = args.margin
        x = cx - r - m
        y = cy - r - m
        w = (r + m) * 2
        h = (r + m) * 2
        print(f"  {i+1} => ['x'=>{x},'y'=>{y},'width'=>{w},'height'=>{h}],")
        print(f"       center: ({cx}, {cy})  radius: {r}")
    print("─" * 62)
    print("Reorder entries to match meter significance (dial 1 = most significant).")


if __name__ == "__main__":
    main()
