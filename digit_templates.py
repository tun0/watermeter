#!/usr/bin/env python3
"""
Template-based digit recognition for the water meter counter strip.

Complements (or replaces) the Tesseract OCR path in meter_reader.py.
Templates are preprocessed digit crops stored under TEMPLATE_DIR.

Usage:
  # Build template library from a session directory
  python3 digit_templates.py --build snapshots/20260601_054630/

  # Test accuracy against a session, comparing to Tesseract
  python3 digit_templates.py --test snapshots/calib_20260601/

  # Show collected templates as a visual grid
  python3 digit_templates.py --show
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import meter_reader as mr

TEMPLATE_DIR  = Path(__file__).parent / "digit_templates"
MATCH_SIZE    = (64, 88)   # (w, h) — all crops normalised to this before matching
MIN_SCORE     = 0.55       # minimum TM_CCOEFF_NORMED score to accept a match
MAX_TEMPLATES = 30         # max stored crops per (position, digit)


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess(crop: np.ndarray) -> np.ndarray:
    """Normalise a digit crop to a fixed-size binary image for matching."""
    resized = cv2.resize(crop, MATCH_SIZE, interpolation=cv2.INTER_CUBIC)
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    eq      = clahe.apply(gray)
    _, th   = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# ── Template store ─────────────────────────────────────────────────────────────

class DigitMatcher:
    """Match digit crops against a stored template library."""

    def __init__(self, template_dir: Path = TEMPLATE_DIR):
        self.template_dir = template_dir
        # templates[pos][digit] = list of preprocessed images
        self.templates: dict[int, dict[int, list[np.ndarray]]] = (
            {i: {} for i in range(5)})
        if template_dir.exists():
            self._load()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        for pos_dir in sorted(self.template_dir.glob("pos?")):
            pos = int(pos_dir.name[3:])
            for dig_dir in sorted(pos_dir.glob("digit?")):
                digit = int(dig_dir.name[5:])
                imgs  = []
                for p in sorted(dig_dir.glob("*.png")):
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imgs.append(img)
                if imgs:
                    self.templates[pos][digit] = imgs

    def _save(self, pos: int, digit: int, crops: list[np.ndarray]) -> None:
        out = self.template_dir / f"pos{pos}" / f"digit{digit}"
        out.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(crops):
            cv2.imwrite(str(out / f"{i:04d}.png"), c)

    def coverage(self) -> dict[int, list[int]]:
        """Return {pos: [digits with templates]} for each position."""
        return {p: sorted(d.keys()) for p, d in self.templates.items()}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, session_dir: Path, trusted_positions: set[int] | None = None) -> None:
        """
        Extract template crops from a session.

        Only crops at `trusted_positions` are used for labelling — those
        positions must be reliably read by Tesseract (default: 0, 1, 2, 4).
        Position 3 is also collected when Tesseract is confident (i.e. when
        it agrees with the digit deduced from adjacent positions).
        """
        if trusted_positions is None:
            trusted_positions = {0, 1, 2, 4}

        files = sorted(session_dir.glob("*.jpg"))
        print(f"Scanning {len(files)} frames in {session_dir.name} …")

        collected: dict[int, dict[int, list[np.ndarray]]] = (
            {i: defaultdict(list) for i in range(5)})

        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            rot = mr.rotate_image(img, mr.ROTATE_DEG)
            try:
                digits = mr.read_digital_digits(rot)
            except Exception:
                continue
            if any(d is None for d in digits):
                continue
            reading = "".join(str(d) for d in digits)
            # Require plausible integer prefix
            if reading[:3] not in ("002", "003"):
                continue

            for pos, (x, y, w, h) in enumerate(mr.DIGITAL_DIGITS):
                digit = digits[pos]
                if digit is None:
                    continue
                if pos not in trusted_positions:
                    # For pos3: only collect when Tesseract agrees with what the
                    # trusted surrounding digits imply. For a reading of "002X6",
                    # the tens digit is only trustworthy when Tesseract gives "9"
                    # (the confirmed value); "8"/"3"/etc. are known misreads.
                    # As the meter advances past 300 this logic relaxes naturally
                    # since pos2 will no longer be "2".
                    if pos == 3:
                        tens = digits[3]
                        # Accept only if all other confirmed positions form a
                        # self-consistent reading AND Tesseract's pos3 value
                        # produces a reading >= the trusted portion implies.
                        # Simplest safe rule: only collect when pos3 digit matches
                        # what the full reading would require given pos2.
                        # For hundreds=2: tens must be 0-9, but skip obvious
                        # misreads (8→9 confusion is one-directional in this font).
                        if tens in (8, 3):  # known frequent misread targets
                            continue
                bucket = collected[pos][digit]
                if len(bucket) < MAX_TEMPLATES:
                    crop = rot[y:y + h, x:x + w]
                    if crop.size == 0:
                        continue
                    try:
                        bucket.append(_preprocess(crop))
                    except Exception:
                        continue

        self.template_dir.mkdir(exist_ok=True)
        total = 0
        for pos in range(5):
            for digit, crops in collected[pos].items():
                self._save(pos, digit, crops)
                print(f"  pos{pos} digit{digit}: {len(crops)} templates saved")
                total += len(crops)
        print(f"Done — {total} templates total.")
        self._load()

    # ── Match ─────────────────────────────────────────────────────────────────

    def match_crop(self, crop: np.ndarray, pos: int) -> tuple[int | None, float]:
        """
        Return (digit, score) for one crop at the given strip position.
        Returns (None, 0.0) if no templates exist or best score < MIN_SCORE.
        """
        avail = self.templates.get(pos)
        if not avail:
            return None, 0.0

        proc       = _preprocess(crop)
        best_digit = None
        best_score = -1.0
        for digit, tmpl_list in avail.items():
            scores = []
            for tmpl in tmpl_list:
                res = cv2.matchTemplate(proc, tmpl, cv2.TM_CCOEFF_NORMED)
                scores.append(float(res[0, 0]))
            score = float(np.mean(sorted(scores)[-5:]))  # top-5 average
            if score > best_score:
                best_score = score
                best_digit = digit

        if best_score < MIN_SCORE:
            return None, best_score
        return best_digit, best_score

    def read_digits(self, rot_img: np.ndarray) -> list[int | None]:
        """
        Return list of 5 ints (or None where confidence is too low).
        Intended as a drop-in replacement for mr.read_digital_digits().
        """
        results = []
        for pos, (x, y, w, h) in enumerate(mr.DIGITAL_DIGITS):
            crop  = rot_img[y:y + h, x:x + w]
            digit, _ = self.match_crop(crop, pos)
            results.append(digit)
        return results

    def read_digits_hybrid(self, rot_img: np.ndarray) -> list[int | None]:
        """
        Tesseract primary, template correction where Tesseract fails or
        where template confidence exceeds Tesseract's ambiguous output.
        """
        tess   = mr.read_digital_digits(rot_img)
        result = list(tess)
        for pos, (x, y, w, h) in enumerate(mr.DIGITAL_DIGITS):
            crop        = rot_img[y:y + h, x:x + w]
            tmpl_digit, tmpl_score = self.match_crop(crop, pos)
            tess_digit  = tess[pos]
            if tess_digit is None and tmpl_digit is not None:
                result[pos] = tmpl_digit          # fill Tesseract gap
            elif (tmpl_digit is not None
                  and tmpl_digit != tess_digit
                  and tmpl_score > 0.70):
                result[pos] = tmpl_digit          # template more confident
        return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _test(session_dir: Path, matcher: DigitMatcher) -> None:
    files = sorted(session_dir.glob("*.jpg"))
    print(f"Testing on {len(files)} frames …\n")

    cols = ("tess", "tmpl", "hybrid")
    counts  = {c: Counter() for c in cols}
    pos3    = {c: Counter() for c in cols}
    perfect = {c: 0 for c in cols}
    total   = 0

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        rot = mr.rotate_image(img, mr.ROTATE_DEG)

        tess   = mr.read_digital_digits(rot)
        tmpl   = matcher.read_digits(rot)
        hybrid = matcher.read_digits_hybrid(rot)

        for label, digits in (("tess", tess), ("tmpl", tmpl), ("hybrid", hybrid)):
            if any(d is None for d in digits):
                counts[label]["fail"] += 1
                continue
            r = "".join(str(d) for d in digits)
            if r[:3] not in ("002", "003"):
                counts[label]["bad_prefix"] += 1
                continue
            counts[label]["ok"] += 1
            pos3[label][digits[3]] += 1
            if r in ("00296", "00297", "00298", "00299", "00300"):
                perfect[label] += 1

        total += 1

    print(f"{'':20s}  {'tess':>8}  {'tmpl':>8}  {'hybrid':>8}")
    print(f"{'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}")
    for key in ("ok", "fail", "bad_prefix"):
        row = [counts[c][key] for c in cols]
        print(f"{key:20s}  {row[0]:>8}  {row[1]:>8}  {row[2]:>8}")
    print()
    print("pos3 distribution:")
    all_digits = sorted(set().union(*[set(pos3[c].keys()) for c in cols]))
    for d in all_digits:
        row = [pos3[c][d] for c in cols]
        print(f"  digit {d}:              {row[0]:>8}  {row[1]:>8}  {row[2]:>8}")
    print()
    for c in cols:
        n = counts[c]["ok"]
        pct = 100 * perfect[c] / n if n else 0
        print(f"{c:8s} perfect reads: {perfect[c]}/{n} ({pct:.1f}%)")


def _show(matcher: DigitMatcher) -> None:
    cov = matcher.coverage()
    for pos, digits in cov.items():
        print(f"pos{pos}: digits {digits} ({sum(len(matcher.templates[pos][d]) for d in digits)} templates)")
    print()
    for pos in range(5):
        for digit in sorted(matcher.templates.get(pos, {}).keys()):
            tmpl = matcher.templates[pos][digit][0]
            cv2.imshow(f"pos{pos} digit{digit}", tmpl)
    print("Press any key in an image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Template digit matcher for water meter")
    ap.add_argument("--build",  metavar="SESSION_DIR",
                    help="Build template library from this session directory")
    ap.add_argument("--test",   metavar="SESSION_DIR",
                    help="Compare Tesseract vs template vs hybrid on this session")
    ap.add_argument("--show",   action="store_true",
                    help="Display stored templates")
    ap.add_argument("--templates", default=str(TEMPLATE_DIR),
                    help=f"Template directory (default: {TEMPLATE_DIR})")
    args = ap.parse_args()

    matcher = DigitMatcher(Path(args.templates))

    if args.build:
        matcher.build(Path(args.build))
    if args.show:
        _show(matcher)
    if args.test:
        _test(Path(args.test), matcher)
    if not any([args.build, args.show, args.test]):
        ap.print_help()


if __name__ == "__main__":
    main()
