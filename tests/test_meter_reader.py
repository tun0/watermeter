"""
Tests for meter_reader.py logic.

All pure-logic functions are tested with injected values; no env var dependency
at test time (conftest.py sets the required env vars before import).

The .env.dist completeness test verifies that every _env() call in the module
has a corresponding entry in .env.dist, and that loading .env.dist values (plus
a dummy CAM_SNAPSHOT_URL) is sufficient to initialise the module.
"""
import os
import re
import time
from pathlib import Path

import pytest

import meter_reader as mr

# Zero offsets loaded from conftest env (126, 270, 270, 0).
Z = [126.0, 270.0, 270.0, 0.0]


# ── corrected_angle ───────────────────────────────────────────────────────────

def test_corrected_angle_zero():
    assert mr.corrected_angle(126.0, 126.0) == pytest.approx(0.0)

def test_corrected_angle_basic():
    assert mr.corrected_angle(216.0, 126.0) == pytest.approx(90.0)

def test_corrected_angle_wraparound():
    assert mr.corrected_angle(100.0, 126.0) == pytest.approx(334.0)

def test_corrected_angle_full_wrap():
    assert mr.corrected_angle(126.0 + 360.0, 126.0) == pytest.approx(0.0)


# ── corrected_digit ───────────────────────────────────────────────────────────

def test_corrected_digit_zero():
    assert mr.corrected_digit(0.0) == 0

def test_corrected_digit_mid():
    assert mr.corrected_digit(90.0) == 2   # 90/36 = 2.5 → int = 2

def test_corrected_digit_nine():
    assert mr.corrected_digit(324.0) == 9

def test_corrected_digit_wraps():
    assert mr.corrected_digit(360.0) == 0


# ── dial_influenced_digit ─────────────────────────────────────────────────────
# Use zero_offsets = [0, 0, 0, 0] for simple cases by overriding the module attr.

@pytest.fixture()
def zero_offsets(monkeypatch):
    monkeypatch.setattr(mr, "DIAL_ZERO_OFFSETS", [0.0, 0.0, 0.0, 0.0])


def test_dial_influence_no_correction_mid(zero_offsets):
    # A0 at digit 3 (108°), A1 at 180° (sub=0.5, mid-revolution) — no correction
    angles = [108.0, 180.0, 180.0, 180.0]
    assert mr.dial_influenced_digit(0, angles) == 3

def test_dial_influence_advance_when_both_confirm(zero_offsets):
    # A0 sub_frac=0.94 (near upper boundary of digit 3), A1 corrected=18° (5% → below LOW=0.15)
    # → should advance to digit 4
    angles = [3 * 36 + 0.94 * 36, 18.0, 0.0, 0.0]
    assert mr.dial_influenced_digit(0, angles) == 4

def test_dial_influence_no_advance_when_driver_not_past_zero(zero_offsets):
    # A0 sub_frac=0.94 but A1 corrected=300° (83% → above LOW) — driver hasn't passed zero
    angles = [3 * 36 + 0.94 * 36, 300.0, 0.0, 0.0]
    assert mr.dial_influenced_digit(0, angles) == 3

def test_dial_influence_retreat_when_both_confirm(zero_offsets):
    # A0 sub_frac=0.05 (just crossed boundary into digit 4), A1 corrected=342° (95% → above HIGH)
    # → driver hasn't completed revolution, A0 hasn't actually crossed → retreat to digit 3
    angles = [4 * 36 + 0.05 * 36, 342.0, 0.0, 0.0]
    assert mr.dial_influenced_digit(0, angles) == 3

def test_dial_influence_no_retreat_when_driver_past_zero(zero_offsets):
    # A0 sub_frac=0.05 but A1 corrected=20° (5.6% → below HIGH) — driver already past zero
    angles = [4 * 36 + 0.05 * 36, 20.0, 0.0, 0.0]
    assert mr.dial_influenced_digit(0, angles) == 4

def test_dial_influence_last_dial_no_driver(zero_offsets):
    # A3 has no driver (index 4 out of range) — returns direct digit
    angles = [0.0, 0.0, 0.0, 108.0]
    assert mr.dial_influenced_digit(3, angles) == 3

def test_dial_influence_none_raw_returns_zero(zero_offsets):
    angles = [None, 180.0, 180.0, 180.0]
    assert mr.dial_influenced_digit(0, angles) == 0

def test_dial_influence_none_driver_no_correction(zero_offsets):
    # Driver is None → no correction, return direct digit
    angles = [3 * 36 + 0.94 * 36, None, 0.0, 0.0]
    assert mr.dial_influenced_digit(0, angles) == 3

def test_dial_influence_cascade_a1_uses_a2(zero_offsets):
    # Cascade: A1 boundary correction driven by A2
    angles = [180.0, 2 * 36 + 0.94 * 36, 18.0, 0.0]
    assert mr.dial_influenced_digit(1, angles) == 3


# ── assemble_reading ──────────────────────────────────────────────────────────

def test_assemble_reading_raises_on_none_digit():
    with pytest.raises(ValueError, match="OCR failure"):
        mr.assemble_reading([None, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0])

def test_assemble_reading_raises_on_none_angle():
    with pytest.raises(ValueError, match="Needle detection"):
        mr.assemble_reading([0, 0, 0, 0, 0], [None, 0.0, 0.0, 0.0])

def test_assemble_reading_integer_part():
    # With all dials at zero_offset → corrected 0 → digit 0 → fractional 0
    raw = Z  # each dial at its zero_offset → corrected angle 0 → digit 0
    reading = mr.assemble_reading([0, 0, 2, 9, 7], raw)
    assert int(reading) == 297

def test_assemble_reading_fractional():
    # A0 at zero_offset+36 → corrected 36 → digit 1 → contributes 0.1
    # All others at zero_offset → digit 0
    raw = [Z[0] + 36.0, Z[1], Z[2], Z[3]]
    reading = mr.assemble_reading([0, 0, 0, 0, 0], raw)
    assert reading == pytest.approx(0.1, abs=1e-4)

def test_assemble_reading_carry_from_inner_dial_advance():
    # A1 is at digit 9, sub=0.96 (> DIAL_INFLUENCE_HIGH): dial_influenced_digit advances it to 0.
    # A2 is near angle 0 (driver_sub=0.003 < DIAL_INFLUENCE_LOW): this is what triggers the advance.
    # A0 is at digit 5, sub=0.5: its own advance condition does NOT fire (sub < DIAL_INFLUENCE_HIGH).
    #
    # Without carry propagation: reading = 310.5006 (A1 drops 9→0, A0 stays at 5).
    # With carry propagation:    reading = 310.6006 (A1 advance carries into A0: 5→6).
    raw = [
        Z[0] + 5.5 * 36,   # A0: corrected=198°, digit=5, sub=0.5
        Z[1] + 9.96 * 36,  # A1: corrected=358.56°, digit=9, sub=0.96 → will be advanced to 0
        Z[2] + 1.0,         # A2: corrected=1°, digit=0, driver_sub=0.003 → triggers A1 advance
        Z[3] + 6.0 * 36,   # A3: corrected=216°, digit=6
    ]
    assert mr.assemble_reading([0, 0, 3, 1, 0], raw) == pytest.approx(310.6006, abs=1e-4)


# ── corrected_fraction ────────────────────────────────────────────────────────

def test_corrected_fraction_all_zeros():
    assert mr.corrected_fraction(Z) == pytest.approx(0.0)

def test_corrected_fraction_none_returns_none():
    assert mr.corrected_fraction([None, Z[1], Z[2], Z[3]]) is None

def test_corrected_fraction_9000():
    # A0 corrected digit 9 (raw = Z[0] + 9*36 = 126+324=450 % 360 = 90)
    # A1,A2,A3 at zero → digits 0
    raw = [Z[0] + 9 * 36, Z[1], Z[2], Z[3]]
    assert mr.corrected_fraction(raw) == pytest.approx(0.9, abs=1e-4)


# ── rollover_coverage ─────────────────────────────────────────────────────────

def _state(last: float, last_frac_override: float | None = None) -> dict:
    """Helper: state dict with last_reading and optional fractional override."""
    s = {"last_reading": last}
    if last_frac_override is not None:
        # Encode the desired fractional into last_reading
        s["last_reading"] = int(last) + last_frac_override
    return s

def _angles_for_frac(frac: float) -> list[float]:
    """Raw angles that produce a given corrected_fraction (using A0 only, others at 0)."""
    # frac = d0 * 0.1, so d0 = round(frac / 0.1)
    # Use A0 to encode the leading digit; others all at zero offset
    d0 = round(frac * 10) % 10
    return [Z[0] + d0 * 36, Z[1], Z[2], Z[3]]

def test_rollover_in_progress_forces_old_digit():
    # 306→307: during rollover only D4 transitions (D4=6, not 9)
    # last_int=306 → last_digs=[0,0,3,0,6]
    state  = {"last_reading": 306.5}
    angles = _angles_for_frac(0.9)
    result = mr.rollover_coverage([0, 0, 3, 7, 5], angles, state)
    # Only D4 in transitioning; forced to old value
    assert result == [0, 0, 3, 7, 6]

def test_rollover_complete_forces_new_digit():
    # 306.9 → just completed: D4 forced to new value (7)
    state  = {"last_reading": 306.9}
    angles = _angles_for_frac(0.0)
    result = mr.rollover_coverage([0, 0, 3, 0, 5], angles, state)
    assert result == [0, 0, 3, 0, 7]

def test_rollover_in_progress_forces_9_when_was_9():
    # 309.5: D4=9, D3=0; transitioning=[4,3]; during → D4=9, D3=0
    state  = {"last_reading": 309.5}
    angles = _angles_for_frac(0.9)
    result = mr.rollover_coverage([0, 0, 3, 7, 5], angles, state)
    assert result == [0, 0, 3, 0, 9]

def test_rollover_complete_forces_0_when_was_9():
    # 309.9 → just completed: D4=0, D3=1
    state  = {"last_reading": 309.9}
    angles = _angles_for_frac(0.0)
    result = mr.rollover_coverage([0, 0, 3, 7, 5], angles, state)
    assert result == [0, 0, 3, 1, 0]

def test_rollover_mid_reading_no_change():
    state  = {"last_reading": 306.3}
    angles = _angles_for_frac(0.4)
    digital = [0, 0, 3, 0, 5]
    assert mr.rollover_coverage(digital, angles, state) == digital

def test_rollover_no_state_no_change():
    digital = [0, 0, 3, 0, 5]
    assert mr.rollover_coverage(digital, _angles_for_frac(0.9), {}) == digital

def test_rollover_cascade_299_to_300():
    # last_int=299 → last_digs=[0,0,2,9,9]; transitioning=[4,3,2]
    # During: D4=9, D3=9, D2=2; D1 and D0 untouched
    state  = {"last_reading": 299.5}
    angles = _angles_for_frac(0.9)
    result = mr.rollover_coverage([0, 0, 3, 5, 7], angles, state)
    assert result == [0, 0, 2, 9, 9]

def test_rollover_cascade_299_complete():
    # After 299→300: D4=0, D3=0, D2=3; D1 and D0 untouched
    state  = {"last_reading": 299.9}
    angles = _angles_for_frac(0.0)
    result = mr.rollover_coverage([0, 0, 3, 5, 7], angles, state)
    assert result == [0, 0, 3, 0, 0]

def test_rollover_cascade_stops_at_non_nine():
    # last_int=294 → last_digs=[0,0,2,9,4]; transitioning=[4] only (D4=4 ≠ 9)
    state  = {"last_reading": 294.5}
    angles = _angles_for_frac(0.9)
    result = mr.rollover_coverage([0, 0, 2, 9, 7], angles, state)
    # D4 forced to 4; all others unchanged
    assert result == [0, 0, 2, 9, 4]

def test_rollover_none_angles_no_change():
    state  = {"last_reading": 306.5}
    angles = [None, Z[1], Z[2], Z[3]]
    digital = [0, 0, 3, 0, 5]
    result = mr.rollover_coverage(digital, angles, state)
    assert result == digital


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_first_reading():
    ok, reason = mr.validate(100.0, {})
    assert ok
    assert reason == "first reading"

def test_validate_ok():
    ok, _ = mr.validate(100.02, {"last_reading": 100.0})
    assert ok

def test_validate_at_max_step():
    ok, _ = mr.validate(100.0 + mr.MAX_STEP, {"last_reading": 100.0})
    assert ok

def test_validate_jump_too_large():
    ok, reason = mr.validate(100.0 + mr.MAX_STEP + 0.001, {"last_reading": 100.0})
    assert not ok
    assert "exceeds allowed" in reason

def test_validate_decrease_within_jitter():
    ok, _ = mr.validate(100.0 - mr.JITTER_TOLERANCE / 2, {"last_reading": 100.0})
    assert ok

def test_validate_decrease_exceeds_jitter():
    ok, reason = mr.validate(100.0 - mr.JITTER_TOLERANCE - 0.001, {"last_reading": 100.0})
    assert not ok
    assert "jitter" in reason

def test_validate_scales_with_elapsed(monkeypatch):
    # Simulate a 30s gap (3 intervals) — allowed should be 3 × MAX_STEP
    old_ts = time.time() - 30
    state  = {"last_reading": 100.0, "last_reading_ts": old_ts}
    allowed = mr.MAX_STEP * 3
    ok, _  = mr.validate(100.0 + allowed, state)
    assert ok

def test_validate_cap_limits_scaled_delta(monkeypatch):
    # Very long gap, but cap prevents accepting unlimited delta
    old_ts = time.time() - 10000
    state  = {"last_reading": 100.0, "last_reading_ts": old_ts}
    ok, reason = mr.validate(100.0 + mr.MAX_DELTA_CAP + 0.001, state)
    assert not ok

def test_validate_no_timestamp_uses_max_step():
    ok, reason = mr.validate(100.0 + mr.MAX_STEP + 0.001,
                             {"last_reading": 100.0})
    assert not ok

def test_validate_at_max_delta_cap_is_rejected():
    # delta == MAX_DELTA_CAP must be rejected; the cap is an exclusive upper bound.
    # Previously delta > allowed (strict) let exactly-cap readings through.
    old_ts = time.time() - 10000
    state  = {"last_reading": 100.0, "last_reading_ts": old_ts}
    ok, reason = mr.validate(100.0 + mr.MAX_DELTA_CAP, state)
    assert not ok
    assert "cap" in reason


# ── corrected_fraction_exit ───────────────────────────────────────────────────

def test_corrected_fraction_exit_stable_near_boundary():
    # A0 at digit 9, sub=0.96: dial_influenced_digit (used by corrected_fraction)
    # advances A0 to 0, dropping the fraction to ~0.0.
    # corrected_fraction_exit uses corrected_digit only — A0 stays at 9 → 0.9.
    a0_near_boundary = Z[0] + 9.96 * 36  # corrected=358.56°, digit=9, sub=0.96
    a1_at_zero       = Z[1]              # corrected=0°, driver_sub=0 → triggers advance
    raw = [a0_near_boundary, a1_at_zero, Z[2], Z[3]]
    assert mr.corrected_fraction(raw)      == pytest.approx(0.0, abs=1e-4)
    assert mr.corrected_fraction_exit(raw) == pytest.approx(0.9, abs=1e-4)

def test_rollover_complete_does_not_fire_on_premature_fraction_advance():
    # A0 sub=0.96 near 9→0: dial_influenced_digit advances A0 prematurely,
    # making corrected_fraction < ROLLOVER_START while A0 hasn't physically crossed.
    # With last_frac >= ROLLOVER_START the "complete" branch must NOT fire.
    a0_near_boundary = Z[0] + 9.96 * 36
    a1_at_zero       = Z[1]
    angles = [a0_near_boundary, a1_at_zero, Z[2], Z[3]]
    state  = {"last_reading": 310.9853}   # last_frac=0.9853 >= ROLLOVER_START
    result = mr.rollover_coverage([0, 0, 3, 1, 0], angles, state)
    assert result == [0, 0, 3, 1, 0]     # no spurious advance to 311


# ── .env.dist completeness ────────────────────────────────────────────────────

def test_env_dist_covers_all_required_vars():
    """Every _env("VAR") call in meter_reader.py must appear in .env.dist."""
    src = Path(__file__).parent.parent / "meter_reader.py"
    env_dist = Path(__file__).parent.parent / ".env.dist"

    required = set(re.findall(r'_env\("([^"]+)"\)', src.read_text()))
    dist_text = env_dist.read_text()
    dist_vars = set(re.findall(r'^#?([A-Z_]+)=', dist_text, re.MULTILINE))

    missing = required - dist_vars
    assert not missing, f"Variables in _env() calls but missing from .env.dist: {missing}"

def test_env_dist_sufficient_to_init(monkeypatch, tmp_path):
    """
    Loading .env.dist values (plus a dummy CAM_SNAPSHOT_URL) must be sufficient
    to import and initialise meter_reader without error.
    """
    env_dist = Path(__file__).parent.parent / ".env.dist"
    dist_vars = {}
    for line in env_dist.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line:
            k, _, v = line.partition('=')
            dist_vars[k.strip()] = v.strip()

    # Provide required site-specific values not set in .env.dist
    dist_vars.setdefault("CAM_SNAPSHOT_URL", "http://test.example.com/")
    dist_vars["STATE_FILE"] = str(tmp_path / "state.json")

    # Apply to environment and reload the module
    env_backup = {k: os.environ.get(k) for k in dist_vars}
    for k, v in dist_vars.items():
        os.environ[k] = v
    try:
        import importlib
        importlib.reload(mr)
    except Exception as e:
        pytest.fail(f"Module failed to initialise with .env.dist values: {e}")
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(mr)  # restore to conftest values


# ── load_state / save_state ───────────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "STATE_FILE", tmp_path / "state.json")
    state = {"last_reading": 123.456, "last_reading_ts": 1234567890.0}
    mr.save_state(state)
    loaded = mr.load_state()
    assert loaded["last_reading"] == pytest.approx(123.456)
    assert loaded["last_reading_ts"] == pytest.approx(1234567890.0)

def test_load_state_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "STATE_FILE", tmp_path / "nonexistent.json")
    assert mr.load_state() == {}
