"""Tests for dispatcher environment lifecycle."""

import pytest

from dead_air.server.dispatcher_environment import DispatcherEnvironment


def test_environment_reset():
    env = DispatcherEnvironment(seed=42)
    obs = env.reset("warmup")
    assert obs["step_number"] == 0
    assert obs["max_steps"] == 80
    assert len(obs["unit_statuses"]) == 6


def test_environment_hold_step():
    env = DispatcherEnvironment(seed=42)
    env.reset()
    obs = env.step({"action_type": "hold"})
    assert obs["step_number"] == 1
    assert any("Hold" in e for e in obs["recent_events"])


def test_environment_dispatch():
    env = DispatcherEnvironment(seed=42)
    env.reset("warmup")
    # Wait for a call to arrive
    for _ in range(20):
        obs = env.step({"action_type": "hold"})
        if obs["active_calls"]:
            break
    if obs["active_calls"]:
        call_id = obs["active_calls"][0]["call_id"]
        obs = env.step({"action_type": "dispatch", "unit_id": 0, "call_id": call_id})
        assert any("Dispatched" in e for e in obs["recent_events"])


def test_environment_log():
    env = DispatcherEnvironment(seed=42)
    env.reset()
    obs = env.step({"action_type": "log", "note": "Test log entry"})
    assert "Test log entry" in obs["dispatch_log"]


def test_environment_verify():
    env = DispatcherEnvironment(seed=42)
    env.reset("warmup")
    # Wait for a call
    for _ in range(20):
        obs = env.step({"action_type": "hold"})
        if obs["active_calls"]:
            break
    if obs["active_calls"]:
        call_id = obs["active_calls"][0]["call_id"]
        obs = env.step({"action_type": "verify", "call_id": call_id})
        assert any("Verified" in e for e in obs["recent_events"])


def test_environment_episode_ends_at_max_steps():
    env = DispatcherEnvironment(seed=42)
    env.reset()
    obs = None
    for _ in range(85):
        obs = env.step({"action_type": "hold"})
        if obs.get("done"):
            break
    assert obs["done"] is True


def test_environment_ground_truth():
    env = DispatcherEnvironment(seed=42)
    env.reset("learning")
    for _ in range(10):
        env.step({"action_type": "hold"})
    gt = env.get_ground_truth()
    assert "calls" in gt
    assert "unit_reliability" in gt


def test_environment_stage():
    env = DispatcherEnvironment(seed=42)
    env.reset()
    obs = env.step({"action_type": "stage", "unit_id": 0, "location_node": 5})
    assert any("Staged" in e for e in obs["recent_events"])


def test_environment_mutual_aid():
    env = DispatcherEnvironment(seed=42)
    env.reset()
    obs = env.step({"action_type": "request_mutual_aid"})
    assert any("Mutual aid" in e for e in obs["recent_events"])
    assert obs["mutual_aid_remaining"] == 1
