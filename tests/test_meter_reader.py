import pytest

import meter_reader as mr

# ── angle_to_digit ────────────────────────────────────────────────────────────

def test_angle_to_digit_boundaries():
    assert mr.angle_to_digit(0.0) == 0
    assert mr.angle_to_digit(35.9) == 0
    assert mr.angle_to_digit(36.0) == 1
    assert mr.angle_to_digit(324.0) == 9
    assert mr.angle_to_digit(359.9) == 9
    assert mr.angle_to_digit(360.0) == 0  # wraps


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_first_reading():
    ok, reason = mr.validate(100.0, {})
    assert ok
    assert reason == "first reading"


def test_validate_ok():
    ok, _ = mr.validate(100.0 + mr.MAX_STEP / 2, {"last_reading": 100.0})
    assert ok


def test_validate_at_max_step():
    ok, _ = mr.validate(100.0 + mr.MAX_STEP, {"last_reading": 100.0})
    assert ok


def test_validate_jump_too_large():
    ok, reason = mr.validate(100.0 + mr.MAX_STEP + 0.001, {"last_reading": 100.0})
    assert not ok
    assert "MAX_STEP" in reason


def test_validate_decrease():
    ok, reason = mr.validate(100.0 - mr.MAX_STEP / 2, {"last_reading": 100.0})
    assert not ok
    assert "decrease" in reason


# ── _dial_fraction ────────────────────────────────────────────────────────────

def test_dial_fraction_none_inputs():
    assert mr._dial_fraction(None, None) is None
    assert mr._dial_fraction(45.0, None) is None
    assert mr._dial_fraction(None, 0.0) is None


def test_dial_fraction_basic():
    assert mr._dial_fraction(90.0, 0.0) == pytest.approx(0.25)


def test_dial_fraction_wraparound():
    assert mr._dial_fraction(10.0, 20.0) == pytest.approx(350.0 / 360.0)


def test_dial_fraction_coincident():
    assert mr._dial_fraction(45.0, 45.0) == pytest.approx(0.0)


# ── assemble_reading ──────────────────────────────────────────────────────────

def test_assemble_reading_raises_on_none_digit():
    with pytest.raises(ValueError, match="OCR failure"):
        mr.assemble_reading([None, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0])


def test_assemble_reading_raises_on_none_angle():
    with pytest.raises(ValueError, match="Needle detection"):
        mr.assemble_reading([0, 0, 0, 0, 0], [None, 0.0, 0.0, 0.0])


def test_assemble_reading_all_zeros():
    # All dials at 0° → analog digits all 0 → fractional = 0 + DIAL_PHASE_CORRECTION
    reading = mr.assemble_reading([0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0])
    assert reading == pytest.approx(mr.DIAL_PHASE_CORRECTION, abs=1e-4)


def test_assemble_reading_integer_from_digital():
    reading = mr.assemble_reading([0, 0, 2, 9, 7], [0.0, 0.0, 0.0, 0.0])
    assert reading == pytest.approx(297 + mr.DIAL_PHASE_CORRECTION, abs=1e-4)


def test_assemble_reading_last_dial_rounds_up():
    # Last dial just past the 0.5 mark of digit 0 → rounds up to 1
    # 0.5 frac of digit 0 = 18°; just above: 18.5°
    reading = mr.assemble_reading([0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 18.5])
    # last analog digit becomes 1 → fractional = 0.0001 + DIAL_PHASE_CORRECTION
    assert reading == pytest.approx(0.0001 + mr.DIAL_PHASE_CORRECTION, abs=1e-4)


# ── correct_gear_lash ─────────────────────────────────────────────────────────

def test_correct_gear_lash_no_snap_mid_reading():
    angles = [180.0, 180.0, 180.0, 180.0]
    assert mr.correct_gear_lash(angles) == angles


def test_correct_gear_lash_none_passthrough():
    angles = [None, 180.0, None, 180.0]
    assert mr.correct_gear_lash(angles) == angles


def test_correct_gear_lash_pass1_high_frac():
    # dial[1] at 10° (digit 0, inside LASH_EXT_DEG window)
    # dial[0] at 250° → digit 6, frac ≈ 0.94 > LASH_HIGH → snap to digit 7 (252°)
    angles = [250.0, 10.0, 180.0, 180.0]
    result = mr.correct_gear_lash(angles)
    assert result[0] == 7 * 36
    assert result[1:] == angles[1:]


def test_correct_gear_lash_pass1_low_frac_core_zone():
    # dial[1] at 5° (digit 0, in_core_zone); dial[0] at 5° → frac ≈ 0.14 < LASH_LOW → snap to 1
    angles = [5.0, 5.0, 180.0, 180.0]
    result = mr.correct_gear_lash(angles)
    assert result[0] == 1 * 36


def test_correct_gear_lash_pass1_no_snap_beyond_ext():
    # dial[1] at 150° — beyond LASH_EXT_DEG (120°) → no trigger
    angles = [250.0, 150.0, 180.0, 180.0]
    result = mr.correct_gear_lash(angles)
    assert result[0] == 250.0


def test_correct_gear_lash_pass2_near_zero():
    # dial[1] at 358° → digit 9, frac ≈ 0.944 >= LASH_NEAR_ZERO → snap to 0°
    # dial[0] at 250° → frac > LASH_HIGH → snap to digit 7
    angles = [250.0, 358.0, 180.0, 180.0]
    result = mr.correct_gear_lash(angles)
    assert result[1] == 0.0
    assert result[0] == 7 * 36


# ── resolve_rollover ──────────────────────────────────────────────────────────

def _rollover_state(last_reading: float, offsets: list) -> dict:
    return {"last_reading": last_reading, "dial_zero_offsets": offsets}


def test_resolve_rollover_no_state():
    digital = [0, 0, 2, 9, 7]
    assert mr.resolve_rollover(digital, [0.0] * 4, {}) == digital


def test_resolve_rollover_correct_up():
    # pos4 (units): last=7, OCR=7 (old), dial just crossed zero → correct to 8
    # zero_offset=10°; angle=15° → frac=(15-10)/360 ≈ 0.014 < _ROLLOVER_BAND=0.15
    state = _rollover_state(297.5, [10.0, 20.0, None, None])
    result = mr.resolve_rollover([0, 0, 2, 9, 7], [15.0, 0.0, 0.0, 0.0], state)
    assert result[4] == 8


def test_resolve_rollover_correct_down():
    # pos4 (units): last=7, OCR=8 (new), dial hasn't crossed zero → correct to 7
    # zero_offset=10°; angle=330° → frac=(330-10)/360 ≈ 0.889 > 1-0.15=0.85
    state = _rollover_state(297.5, [10.0, 20.0, None, None])
    result = mr.resolve_rollover([0, 0, 2, 9, 8], [330.0, 0.0, 0.0, 0.0], state)
    assert result[4] == 7


def test_resolve_rollover_no_change_mid_digit():
    # Dial fraction is mid-range — no rollover ambiguity
    state = _rollover_state(297.5, [10.0, 20.0, None, None])
    digital = [0, 0, 2, 9, 7]
    result = mr.resolve_rollover(digital, [180.0, 0.0, 0.0, 0.0], state)
    assert result == digital


def test_resolve_rollover_corrects_garbled_digit():
    # OCR reads 9 for units position during 7→8 transition (not in {7,8}) — dial
    # confirms the rollover has passed → must still correct to 8.
    state = _rollover_state(297.5, [10.0, 20.0, None, None])
    result = mr.resolve_rollover([0, 0, 2, 9, 9], [15.0, 0.0, 0.0, 0.0], state)
    assert result[4] == 8


def test_resolve_rollover_corrects_none_digit():
    # OCR returns None for units during transition — dial is past zero → fill with expected_new.
    state = _rollover_state(297.5, [10.0, 20.0, None, None])
    result = mr.resolve_rollover([0, 0, 2, 9, None], [15.0, 0.0, 0.0, 0.0], state)
    assert result[4] == 8


def test_resolve_rollover_works_when_other_digit_is_none():
    # A None in a non-calibrated position must not block correction of calibrated ones.
    state = _rollover_state(297.5, [10.0, 20.0, None, None])
    result = mr.resolve_rollover([None, 0, 2, 9, 7], [15.0, 0.0, 0.0, 0.0], state)
    assert result[4] == 8  # units corrected despite None in ten-thousands


# ── load_state / save_state ───────────────────────────────────────────────────

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "STATE_FILE", tmp_path / "state.json")
    state = {"last_reading": 123.456, "dial_zero_offsets": [1.0, 2.0, None, None]}
    mr.save_state(state)
    loaded = mr.load_state()
    assert loaded["last_reading"] == pytest.approx(123.456)
    assert loaded["dial_zero_offsets"] == [1.0, 2.0, None, None]


def test_load_state_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "STATE_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(mr, "INITIAL_VALUE", None)
    assert mr.load_state() == {}


def test_load_state_uses_initial_value(tmp_path, monkeypatch):
    monkeypatch.setattr(mr, "STATE_FILE", tmp_path / "nonexistent.json")
    monkeypatch.setattr(mr, "INITIAL_VALUE", 297.1234)
    state = mr.load_state()
    assert state["last_reading"] == pytest.approx(297.1234)
