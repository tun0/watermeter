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

Dial markings on A0, A1 and A2 cannot be trusted — their mounting angle is
arbitrary, so raw angles carry no absolute meaning. A3 is the exception: its
markings can be treated as absolute.


Calibration
-----------

Because A0, A1 and A2 markings are unreliable, each must be calibrated to a
known zero position. A "corrected angle" is the raw angle adjusted by the
dial's calibrated zero offset. All fractions and digit interpretations below
are computed from corrected angles.

A3 requires no calibration.


Rollover
--------

Each increment of D4 (e.g. N → N+1) is accompanied by one full revolution of
A0. During this revolution, the digit drum transitions mechanically and the
digital display becomes temporarily illegible.

  Rollover transition starts: corrected fraction = .9000  (A0 physically at ≈ 90°, showing corrected digit 9)
  Rollover transition ends:   corrected fraction = .0000

During the window, A0 advances exactly 36° (one digit, 9 → 0) and A1 completes
exactly one full revolution, returning to the same physical position it started at.

Assumption: all rollovers behave identically.


Empirical observations
----------------------

Confirmed on rollovers 295→296, 299→300 and 306→307:

  Rollover start: A0 raw ≈  90°,  A1 raw ≈ 270°,  A2 raw ≈ 266°
  Rollover end:   A0 raw ≈ 127°,  A1 raw ≈ 270°,  A2 raw ≈ 266°

Zero offsets (physical properties of the meter dials):
  A0 zero_offset = 126°
  A1 zero_offset = 270°
  A2 zero_offset = 270°

Measured values are within a few degrees of these. The clean numbers are
intentional: this is a refurbished meter with physically displaced dials, and
small deviations from the true mechanical zero are acceptable because the
dial influence from less-significant dials shifts detection zones accordingly.


Dial influence
--------------

The corrected value of A(n) shifts the digit interpretation boundary of A(n-1).
This accounts for mechanical play in the gear train.

Example (all angles corrected):
  A3 at   0° → A2 corrected angles  18° –  40° map to digit 1
  A3 at 180° → A2 corrected angles  32° –  64° map to digit 1


Safeguards
----------

Readings are validated against the previous sample before being accepted.

Primary rate limit: the meter reading must not increase by more than 0.05 m³
per 10-second interval. This reflects the maximum realistic flow rate.

If snapshots were missed (gap > 10s between samples), the limit scales linearly
with elapsed time:

  max_delta = RATE_LIMIT_PER_INTERVAL * (elapsed_seconds / INTERVAL_SECONDS)

This scaled delta is capped by a secondary maximum to avoid accepting arbitrarily
large jumps after long outages:

  max_delta = min(max_delta, MAX_DELTA_CAP)

Both the per-interval rate limit (0.05 m³) and the secondary cap are configured
values, not hard-coded. The cap can be expressed as either a maximum value
difference or a maximum time gap (whichever is more intuitive — the implementation
should support at least one).

A reading that exceeds the allowed delta is rejected; the previous reading is
retained and the event is logged.

Small decreases are accepted: analog dials can settle slightly backward after
flow stops, producing readings a few millilitres below the previous value.
Decreases within a configured jitter tolerance are accepted and reported as
zero flow. Larger decreases are rejected as errors.

State persists two fields across samples: last_reading (the last accepted
value) and last_reading_ts (its Unix timestamp). Both are required for the
scaled delta check and flow rate calculation.


Goals
-----

1. Flow rate accuracy is the primary concern; absolute reading accuracy is secondary.
2. Cover the illegibility gap during every digit rollover (D4, D3, D2, D1, D0).
3. During the gap, use analog dial positions to infer the correct digit value.


Implementation plan
-------------------

General: all tunable parameters (zero offsets, crop coordinates, dial regions,
rate limits, thresholds) live in configuration, not in logic code. Logic code
only contains algorithms.

Configuration is supplied via environment variables. .env.dist is the canonical
reference. Two categories:

  Required with defaults: variables that have sensible shared defaults (dial
  offsets, thresholds, intervals). .env.dist lists the actual default values.
  The script fails explicitly at startup if they are absent.

  Required without defaults: variables that are inherently site-specific (camera
  URL, HA token). .env.dist provides a commented-out dummy value as a reminder.
  The script fails explicitly at startup if absent.

The script never provides hardcoded fallbacks — absence always means failure.

Test coverage is required for all logic: corrected angle primitives, dial
influence (including boundary and cascade cases), rollover detection, and
safeguard validation. Configuration-dependent behaviour must be testable by
injecting values directly without relying on environment variables in tests.
Tests must also verify that supplying the default values from .env.dist (plus
any required site-specific vars) is sufficient for the script to initialise
correctly — ensuring .env.dist is always complete and accurate.

Step 1 — Corrected angle primitives                                  [DONE]
  DIAL_ZERO_OFFSETS config value (env var).
  corrected_angle(raw, zero_offset) → (raw - zero_offset) % 360
  corrected_digit(corrected_angle) → int(corrected_angle / 36) % 10

Step 2 — Rewrite assemble_reading                                    [DONE]
  Uses corrected digits from step 1 instead of raw angles.
  Removed DIAL_PHASE_CORRECTION and old rounding helpers.

Step 3 — Implement dial influence                                     [DONE]
  Replaced correct_gear_lash with dial_influenced_digit(n, raw_angles).
  The rule: A(n) reads as (k+1)%10 only if sub_frac(n) > DIAL_INFLUENCE_HIGH
  AND corrected(n+1)/360 < DIAL_INFLUENCE_LOW (driver just passed 0).
  Symmetric rule prevents premature crossing detection.

Step 4 — Implement rollover coverage                                  [DONE]
  Replaced _apply_rollover_bridge, resolve_rollover, calibrate_from_rollover
  with rollover_coverage(digital, raw_angles, state).
  Transitioning positions: D4 always; D3/D2/... cascade only if ALL
  less-significant positions held 9 (causing a carry into this position).
  During rollover: force to old value (last_digs[pos]).
  After rollover: force to new value ((last_digs[pos]+1) % 10).

Step 5 — Implement scaled safeguards                                  [DONE]
  validate() scales allowed delta by elapsed time, capped at MAX_DELTA_CAP.
  Jitter tolerance allows small backward movement.
  All thresholds are configured values (env vars).

Step 6 — Clean up + config externalisation                           [DONE]
  All tunable values moved to environment variables; no hardcoded fallbacks.
  Canonical defaults documented in .env.dist.
  Removed dead code: DIAL_PHASE_CORRECTION, ROLLOVER_BRIDGE_THRESHOLD,
  gear-lash constants, calibrate_from_rollover, resolve_rollover,
  _apply_rollover_bridge, dial_zero_offsets state field.

Step 7 — Test suite                                                   [DONE]
  48 tests covering all logic functions, rollover cascade cases, scaled
  safeguards, and .env.dist completeness verification.
