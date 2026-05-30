#!/usr/bin/env python3
"""
Annotate collected snapshots with detected values for visual validation.

Writes annotated copies to snapshots/<session>/annotated/
showing: final reading, digital digit boxes+OCR values, dial circles+needles+digits.

Also generates gallery.html in the session directory.

Usage:
  python3 annotate_snapshots.py snapshots/20260529_102212/
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

# Import pipeline from meter_reader
sys.path.insert(0, str(Path(__file__).parent))
from meter_reader import (
    ANALOG_DIALS, DIGITAL_DIGITS, LASH_HIGH, LASH_LOW, MAX_STEP,
    rotate_image, read_digital_digits, read_analog_dials,
    correct_gear_lash, assemble_reading, angle_to_digit,
    ROTATE_DEG,
)

_LAST_DIAL_IDX = len(ANALOG_DIALS) - 1


def _last_dial_digit(angle: float) -> int:
    """Round-to-nearest for the least significant dial, but 9 never wraps to 0."""
    d = int(angle / 36.0) % 10
    return d if d == 9 or (angle / 36.0) % 1.0 < 0.5 else (d + 1) % 10


def annotate_image(img: np.ndarray, filename: str = "",
                   min_reading: float | None = None) -> tuple[str, np.ndarray, float | None]:
    """
    Returns (reading_str, annotated_img, used_float).

    used_float is None on error.  When the assembled reading would drop below
    min_reading it is clamped to min_reading and the title bar turns orange.
    """
    rotated    = rotate_image(img, ROTATE_DEG)
    digital    = read_digital_digits(rotated)
    angles_raw = read_analog_dials(rotated)
    angles_cor = correct_gear_lash(angles_raw)

    carried_forward = False
    used_float: float | None = None
    try:
        raw_val = assemble_reading(digital, angles_cor)
        if min_reading is not None and (
                raw_val < min_reading - 0.0005       # decrease not allowed
                or raw_val > min_reading + MAX_STEP  # jump too large (bad OCR)
        ):
            reading      = f"{min_reading:.4f}"
            used_float   = min_reading
            color_title  = (0, 140, 255)   # orange = carried forward
            carried_forward = True
            fresh_str    = f"{raw_val:.4f}"
        else:
            reading     = f"{raw_val:.4f}"
            used_float  = raw_val
            color_title = (0, 200, 0)      # green = new accepted value
            fresh_str   = reading
    except ValueError as e:
        reading     = f"ERR: {e}"
        fresh_str   = reading
        color_title = (0, 0, 255)
        used_float  = min_reading

    out = rotated.copy()

    # ── Title bar ─────────────────────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (30, 30, 30), -1)
    if carried_forward:
        # Show held value in orange, then fresh (rejected) value in dim grey
        cv2.putText(out, reading, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_title, 2)
        (rw, _), _ = cv2.getTextSize(reading, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.putText(out, f" <- {fresh_str}", (10 + rw, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 120, 180), 1)
    else:
        cv2.putText(out, reading, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_title, 2)
    if filename:
        (fw, _), _ = cv2.getTextSize(filename, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(out, filename, (out.shape[1] - fw - 6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # ── Digital digit boxes ────────────────────────────────────────────────────
    for i, (x, y, w, h) in enumerate(DIGITAL_DIGITS):
        d   = digital[i]
        col = (0, 220, 0) if d is not None else (0, 0, 255)
        cv2.rectangle(out, (x, y), (x + w, y + h), col, 2)
        label = str(d) if d is not None else "?"
        cv2.putText(out, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # ── Dial circles and needles ───────────────────────────────────────────────
    for i, (cx, cy, r, _, _) in enumerate(ANALOG_DIALS):
        cv2.circle(out, (cx, cy), r, (200, 140, 0), 2)

        for angle, color, thickness in (
                (angles_raw[i], (140, 140, 140), 2),
                (angles_cor[i], (0, 60, 255),    3)):
            if angle is not None:
                a   = math.radians(angle)
                tip = (int(cx + (r - 10) * math.sin(a)),
                       int(cy  - (r - 10) * math.cos(a)))
                cv2.line(out, (cx, cy), tip, color, thickness, cv2.LINE_AA)

        if angles_cor[i] is not None:
            d_val = (_last_dial_digit(angles_cor[i]) if i == _LAST_DIAL_IDX
                     else angle_to_digit(angles_cor[i]))
        else:
            d_val = None
        d_str = str(d_val) if d_val is not None else "?"
        was_corrected = (angles_raw[i] != angles_cor[i]
                         and angles_raw[i] is not None
                         and angles_cor[i] is not None)
        label_col = (0, 220, 255) if was_corrected else (255, 220, 0)
        cv2.putText(out, d_str, (cx - 9, cy + 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, label_col, 2)

        if angles_raw[i] is not None:
            cv2.putText(out, f"{angles_raw[i]:.0f}", (cx - 15, cy + r + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    return reading, out, used_float


# ── Gallery HTML generation ────────────────────────────────────────────────────
_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Water meter snapshot gallery</title>
<style>
  body {{ background: #111; color: #eee; font-family: monospace; margin: 0; padding: 12px; }}
  h1   {{ font-size: 1rem; color: #aaa; margin: 0 0 12px; }}
  #filter {{ background: #222; color: #eee; border: 1px solid #444; padding: 4px 8px;
            font-family: monospace; width: 260px; margin-bottom: 12px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .card {{ background: #1e1e1e; border: 1px solid #333; border-radius: 4px;
          width: 260px; cursor: pointer; }}
  .card img {{ width: 260px; height: auto; display: block; border-radius: 3px 3px 0 0; }}
  .card .label {{ padding: 4px 6px; font-size: 0.75rem; }}
  .card .reading {{ font-size: 1rem; font-weight: bold; color: #6cf;
                   user-select: text; cursor: text; }}
  .card .fname   {{ color: #777; font-size: 0.65rem;
                   user-select: text; cursor: text; }}
  .card.outlier  {{ border-color: #f66; }}
  .card.outlier .reading {{ color: #f88; }}
  #lb {{ display:none; position:fixed; inset:0; background:rgba(0,0,0,.85);
        align-items:center; justify-content:center; z-index:10; }}
  #lb img {{ max-width:95vw; max-height:90vh; border-radius:4px; cursor:zoom-out; }}
  #lb.open {{ display:flex; }}
  #lb-prev, #lb-next {{
    position:fixed; top:50%; transform:translateY(-50%);
    background:rgba(255,255,255,.12); border:none; color:#fff;
    font-size:2rem; padding:0.4em 0.7em; cursor:pointer; border-radius:4px;
    user-select:none; z-index:11;
  }}
  #lb-prev {{ left:10px; }}
  #lb-next {{ right:10px; }}
  #lb-prev:hover, #lb-next:hover {{ background:rgba(255,255,255,.25); }}
  #lb-caption {{
    position:fixed; bottom:12px; left:50%; transform:translateX(-50%);
    background:rgba(0,0,0,.6); color:#ccc; font-family:monospace;
    font-size:0.8rem; padding:4px 10px; border-radius:4px;
    pointer-events:auto; user-select:text; cursor:text;
  }}
</style>
</head>
<body>
<h1>Water meter snapshots — session {session} &nbsp;|&nbsp;
    <span id="count"></span> images</h1>
<input id="filter" type="text" placeholder="Filter by reading or filename…" oninput="applyFilter()">
<div class="grid" id="grid"></div>

<div id="lb" onclick="closeLb()">
  <button id="lb-prev" onclick="event.stopPropagation(); lbStep(-1)">&#8249;</button>
  <img id="lb-img" src="">
  <button id="lb-next" onclick="event.stopPropagation(); lbStep(+1)">&#8250;</button>
  <div id="lb-caption" onclick="event.stopPropagation()"></div>
</div>

<script>
const images = [
"""

_HTML_TAIL = """\
];

const median = {median};
const MAX_STEP = 0.5;

function buildCards() {{
  const grid = document.getElementById('grid');
  document.getElementById('count').textContent = images.length;
  images.forEach(({{fname, ann, reading}}) => {{
    const outlier = Math.abs(parseFloat(reading) - median) > MAX_STEP;
    const card = document.createElement('div');
    card.className = 'card' + (outlier ? ' outlier' : '');
    card.dataset.reading = reading;
    card.dataset.fname   = fname;
    card.innerHTML = `
      <img src="annotated/${{ann}}" loading="lazy">
      <div class="label">
        <div class="reading">${{reading}}</div>
        <div class="fname">${{fname}}</div>
      </div>`;
    card.querySelector('img').addEventListener('click', e => {{
      e.stopPropagation();
      refreshVisible();
      openLb(visibleCards.indexOf(card));
    }});
    card.querySelector('.label').addEventListener('click', e => e.stopPropagation());
    grid.appendChild(card);
  }});
}}

function applyFilter() {{
  const q = document.getElementById('filter').value.toLowerCase();
  document.querySelectorAll('.card').forEach(c => {{
    const match = c.dataset.reading.includes(q) || c.dataset.fname.includes(q);
    c.style.display = match ? '' : 'none';
  }});
}}

let lbIndex = -1;
let visibleCards = [];

function refreshVisible() {{
  visibleCards = [...document.querySelectorAll('.card')]
    .filter(c => c.style.display !== 'none');
}}

function openLb(idx) {{
  refreshVisible();
  lbIndex = idx;
  const card = visibleCards[lbIndex];
  const img  = card.querySelector('img');
  document.getElementById('lb-img').src = img.src;
  document.getElementById('lb-caption').textContent =
    `${{lbIndex + 1}} / ${{visibleCards.length}}  —  ${{card.dataset.fname}}  —  ${{card.dataset.reading}}`;
  document.getElementById('lb').classList.add('open');
}}

function lbStep(dir) {{
  refreshVisible();
  lbIndex = (lbIndex + dir + visibleCards.length) % visibleCards.length;
  openLb(lbIndex);
}}

function closeLb() {{ document.getElementById('lb').classList.remove('open'); }}

document.addEventListener('keydown', e => {{
  if (!document.getElementById('lb').classList.contains('open')) return;
  if (e.key === 'ArrowRight') lbStep(+1);
  else if (e.key === 'ArrowLeft')  lbStep(-1);
  else if (e.key === 'Escape')     closeLb();
}});

buildCards();
</script>
</body>
</html>
"""


def write_gallery(session: Path, entries: list[dict]) -> Path:
    valid_readings = [float(e["reading"]) for e in entries
                      if not e["reading"].startswith("ERR")]
    med = statistics.median(valid_readings) if valid_readings else 0.0
    session_name = session.name

    rows = ",\n".join(
        f'  {{"fname": {json.dumps(e["fname"])}, "ann": {json.dumps(e["ann"])}, '
        f'"reading": {json.dumps(e["reading"])}}}'
        for e in entries
    )

    html = (_HTML_HEAD.format(session=session_name)
            + rows + "\n"
            + _HTML_TAIL.format(median=f"{med:.4f}"))

    out = session / "gallery.html"
    out.write_text(html)
    return out


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", help="Path to snapshot session directory")
    args = ap.parse_args()

    session = Path(args.session_dir)
    if not session.is_dir():
        sys.exit(f"Not a directory: {session}")

    out_dir = session / "annotated"
    out_dir.mkdir(exist_ok=True)

    files = sorted(session.glob("*.jpg"))
    if not files:
        sys.exit("No .jpg files found")

    print(f"Annotating {len(files)} images → {out_dir}")
    errors   = 0
    entries  = []
    last_val: float | None = None

    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f"  SKIP (unreadable): {f.name}")
            errors += 1
            continue

        reading, ann, used_val = annotate_image(img, filename=f.name,
                                                min_reading=last_val)
        out_path = out_dir / f.name
        cv2.imwrite(str(out_path), ann)

        entries.append({"fname": f.name, "ann": f.name, "reading": reading})
        if used_val is not None:
            last_val = used_val

        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] last: {reading}")

    gallery = write_gallery(session, entries)
    valid   = [float(e["reading"]) for e in entries if not e["reading"].startswith("ERR")]
    med     = statistics.median(valid) if valid else 0.0
    outliers = [e for e in entries
                if not e["reading"].startswith("ERR")
                and abs(float(e["reading"]) - med) > 0.5]

    print(f"Done. {len(files) - errors} annotated, {errors} skipped.")
    print(f"Gallery: {gallery}  (median={med:.4f}, outliers={len(outliers)})")
    for e in outliers:
        print(f"  {e['fname']}: {e['reading']}")


if __name__ == "__main__":
    main()
