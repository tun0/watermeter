WATER METER
===========


Physical
--------

The meter has 5 digital digits and 4 analog dials.

Digital digits, left to right:
  D0 = 10000 m³,  D1 = 1000 m³,  D2 = 100 m³,  D3 = 10 m³,  D4 = 1 m³

Analog dials, right to left, all rotating clockwise:
  A0 = 0.1 m³,  A1 = 0.01 m³,  A2 = 0.001 m³,  A3 = 0.0001 m³

Gear relationship: every time A(n+1) completes one full revolution (360°),
A(n) advances by exactly one digit (36°). All digits and dials are physically
interconnected. Each full revolution of A0 increments D4 by 1.

The dial markings on all four dials are unreliable — their mounting angle is
arbitrary and each must be individually calibrated.


Calibration
-----------

Each dial has a zero_offset: the raw needle angle at which that dial reads
digit 0. All digit computation starts from the corrected angle:

  corrected_angle = (raw_angle - zero_offset) % 360
  digit           = int(corrected_angle / 36) % 10

Calibrated zero offsets:
  A0: 132°   (refined from 310→311 rollover observation, 2026-06-29)
  A1: 270°
  A2: 270°
  A3:   0°

Offsets are determined empirically by observing rollover events: the offset
is the raw needle angle at which the digit boundary aligns with the display
incrementing.

A3's offset matters: its corrected angle is used to improve the digit
reading of A2 (see Inter-dial correction below).


Reading assembly
----------------

  reading = digital_integer + A0/10 + A1/100 + A2/1000 + A3/10000

The digital integer comes from OCR of the five digit drums. A0–A3 are the
corrected digits from the analog dials.

The analog fraction (right-hand side) is the authoritative source during
rollover windows when the digit drums are in motion and OCR is unreliable.


Inter-dial correction
---------------------

The gear relationship gives a continuous prediction: because A(n+1) drives
A(n) at a 10:1 ratio, A(n+1)'s position within its current revolution
directly predicts where A(n) should be within its current digit:

  expected_sub(n) = corrected_angle(n+1) / 360

where sub (0.0–1.0) is the fractional position within a digit.

Camera-based needle detection has noise. When the expected and observed
sub-positions for A(n) disagree significantly, A(n+1) is the more reliable
source — it spans a much larger arc per unit of A(n)'s digit. Use the
expected sub to select the digit nearest to A(n)'s raw observation.

This handles detection noise at any point in the dial's rotation, not just
near digit boundaries.

Gear phase: the phase offset is the corrected angle of A(n+1) at the moment
A(n) crosses a digit boundary. Empirically confirmed for this meter: the
phase offset is approximately 0° for all dial pairs (A(n) crosses when
A(n+1) is near 0° corrected). No phase correction is therefore needed.


Rollover
--------

When D4 increments (e.g. 310→311), A0 completes one full revolution and the
digit drum is temporarily illegible. The same applies to D3, D2, D1, D0 for
cascade rollovers (e.g. 299→300).

Detection uses the analog fraction F = A0/10 + A1/100 + A2/1000 + A3/10000:

  In progress  (F ≥ ROLLOVER_START, typically 0.9):
    The drum is in transition. The OCR integer is locked to the last accepted
    value; the analog fraction continues to advance normally.

  Complete  (F drops from ≥ ROLLOVER_START to below it):
    A0 has physically crossed digit 0. The OCR integer is incremented by 1.

For cascade rollovers, any digit position that held 9 in the last accepted
reading is also treated as transitioning.


Validation
----------

Each reading is validated against the previous accepted value before being
published.

  Increases up to MAX_STEP per reading interval are accepted. For gaps longer
  than one interval the limit scales with elapsed time, capped at MAX_DELTA_CAP.

  Decreases within JITTER_TOLERANCE are accepted (dials can settle slightly
  backward after flow stops). Larger decreases are rejected as errors.

  Readings outside these bounds are dropped; the previous value is retained
  and the event is logged.

State persists two fields: last_reading and last_reading_ts (Unix timestamp).
Both are required for the scaled delta check and flow rate calculation.


Goals
-----

1. Flow rate accuracy is the primary concern; absolute accuracy is secondary.
2. Cover the illegibility gap during every digit rollover (D4 through D0).
3. During the gap, use analog dial positions to infer the correct integer value.


Configuration
-------------

All tunable parameters are supplied via environment variables. .env.dist is
the canonical reference with all defaults documented.

  Site-specific (no defaults):  CAM_SNAPSHOT_URL, HA_URL, HA_TOKEN
  Shared defaults:               DIAL_ZERO_OFFSETS, ROLLOVER_START,
                                 READING_INTERVAL, MAX_STEP, MAX_DELTA_CAP,
                                 JITTER_TOLERANCE

Logic code contains only algorithms; no hardcoded fallbacks. Absent required
variables cause an explicit failure at startup.
