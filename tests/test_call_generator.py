"""Tests for call generator with panic bias, false alarms, and ghost calls."""

import pytest

from server.call_generator import CallGenerator


def test_call_generator_creates_calls():
    cg = CallGenerator()
    cg.reset("warmup")
    call = cg.generate_call(0, list(range(20)))
    assert call["call_id"] == 1
    assert call["call_type"] in ["cardiac", "trauma", "fire"]


def test_false_alarms_generated():
    cg = CallGenerator()
    cg.reset("learning")
    cg.configure(false_alarm_rate=0.50, panic_range=(1.0, 1.0), ghost_rate=0.0)
    false_count = 0
    for _ in range(20):
        c = cg.generate_call(0, list(range(20)))
        if c["is_false_alarm"]:
            false_count += 1
    assert false_count > 0


def test_panic_modifiers_affect_tone():
    cg = CallGenerator()
    cg.configure(false_alarm_rate=0.0, panic_range=(0.6, 1.5), ghost_rate=0.0)
    tones = set()
    for _ in range(30):
        c = cg.generate_call(0, list(range(20)))
        tones.add(c["caller_tone"])
    assert len(tones) >= 2  # Should see multiple tones


def test_ghost_calls_generated():
    cg = CallGenerator()
    cg.configure(false_alarm_rate=0.0, panic_range=(1.0, 1.0), ghost_rate=1.0)
    c = cg.generate_call(0, list(range(20)))
    assert c["is_ghost"] is True


def test_verify_returns_confidence():
    cg = CallGenerator()
    cg.configure(ghost_rate=0.0)
    c = cg.generate_call(0, list(range(20)))
    conf = cg.verify_call(c["call_id"])
    assert conf in ["high", "medium", "low"]


def test_call_tick_increments_elapsed():
    cg = CallGenerator()
    cg.reset()
    c = cg.generate_call(0, list(range(20)))
    cg.tick(5)
    assert c["time_elapsed"] == 5


def test_adversarial_bias_affects_location():
    cg = CallGenerator()
    cg.reset()
    # Bias heavily toward hills
    calls = []
    for _ in range(20):
        c = cg.generate_call(0, list(range(20)), adversarial_bias={"hills": 5.0})
        calls.append(c)
    hills_calls = sum(1 for c in calls if 8 <= c["location"] <= 11)
    assert hills_calls >= 2  # Should see more hills calls
