"""Tests for OpenEnv server connection and end-to-end episode."""

import pytest

from server.app import app
from server.dispatcher_environment import DispatcherEnvironment


def test_server_app_exists():
    assert app is not None


def test_environment_can_reset():
    env = DispatcherEnvironment(seed=42)
    obs = env.reset("warmup")
    assert obs["step_number"] == 0


def test_environment_full_episode():
    env = DispatcherEnvironment(seed=42)
    obs = env.reset("warmup")
    done = False
    steps = 0
    while not done and steps < 100:
        action = {"action_type": "hold"}
        obs = env.step(action)
        done = obs.get("done", False)
        steps += 1
    assert steps <= 85  # Should end at MAX_STEPS=80


def test_environment_with_dispatch():
    env = DispatcherEnvironment(seed=42)
    env.reset("warmup")
    # Run until we see a call
    obs = None
    for _ in range(20):
        obs = env.step({"action_type": "hold"})
        if obs.get("active_calls"):
            break
    if obs and obs.get("active_calls"):
        call = obs["active_calls"][0]
        obs = env.step(
            {"action_type": "dispatch", "unit_id": 0, "call_id": call["call_id"]}
        )
        assert any("Dispatched" in e for e in obs["recent_events"])


def test_curriculum_updates():
    env = DispatcherEnvironment(seed=42)
    env.reset("curriculum")
    # Simulate some episodes with rewards
    for _ in range(15):
        env.curriculum.record_reward(0.7)
    new_phase = env.curriculum.update_phase()
    assert new_phase in ["warmup", "learning", "advanced", "expert"]


def test_adversarial_tracks_weaknesses():
    env = DispatcherEnvironment(seed=42)
    env.reset("warmup")
    # Record fake failures
    env.adversarial_designer.record_episode(
        calls=[{"call_type": "cardiac", "zone": "hills", "fatality": True}],
        fatalities=1,
    )
    env.adversarial_designer.record_episode(
        calls=[{"call_type": "cardiac", "zone": "hills", "fatality": True}],
        fatalities=1,
    )
    bias = env.adversarial_designer.get_bias()
    assert "hills" in bias or "cardiac" in bias
