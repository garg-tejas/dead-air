"""Greedy emergency dispatch policy for DispatchR.

Provides a fast, deterministic heuristic dispatcher that can be used as:
- A baseline for evaluation
- A reference policy for cache replay
- An exploration fallback during training
"""

import json
from typing import Any, Dict, List, Optional

from .city_graph import CityGraph
from .constants import DEADLINES, SEVERITY_WEIGHTS

# Singleton city graph for fast distance lookups
_CITY_GRAPH = CityGraph()


def greedy_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy dispatch: closest idle unit to highest-priority pending call.

    Priority ordering: cardiac > trauma > fire > false_alarm.
    Uses graph travel time (not Manhattan distance) for correctness.

    Returns:
        DispatchAction dict. Returns hold() when no pending calls or no idle units.
    """
    active_calls = obs.get("active_calls", [])
    unit_statuses = obs.get("unit_statuses", [])

    if not active_calls:
        return {"action_type": "hold"}

    # Priority mapping: higher number = more urgent
    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}

    pending = [c for c in active_calls if c.get("assigned_unit") is None]
    if not pending:
        return {"action_type": "hold"}

    # Sort pending calls by priority (highest first)
    pending.sort(
        key=lambda c: priority.get(c.get("reported_type", "trauma"), 1),
        reverse=True,
    )

    # Find idle units
    idle = [u for u in unit_statuses if u.get("last_known_status") == "idle"]
    if not idle:
        return {"action_type": "hold"}

    # Greedy assignment: for each call (in priority order), find closest idle unit
    best_action = None
    best_score = -float("inf")

    for call in pending:
        call_priority = priority.get(call.get("reported_type", "trauma"), 1)
        for unit in idle:
            unit_loc = unit.get("last_known_location", 0)
            call_loc = call["location"]
            travel_time = _CITY_GRAPH.travel_time(unit_loc, call_loc)

            # Score: high priority wins; tie-break by shorter travel time
            # Priority dominates: cardiac at 10min beats trauma at 2min
            score = call_priority * 1000 - travel_time

            if score > best_score:
                best_score = score
                best_action = {
                    "action_type": "dispatch",
                    "unit_id": unit["unit_id"],
                    "call_id": call["call_id"],
                }

    return best_action if best_action else {"action_type": "hold"}


def run_greedy_episode(env_ref, max_steps: int = 80) -> List[Dict[str, Any]]:
    """Run a full episode using the greedy policy.

    Returns a list of step records, each containing:
        - step: int
        - obs: dict (the observation BEFORE the action)
        - action: dict (the greedy action taken)
        - events: list[str]
    """
    trajectory = []
    env_ref.reset()

    for step_idx in range(max_steps):
        obs = env_ref._obs
        if obs is None or obs.get("done"):
            break

        action = greedy_action(obs)
        events_text = env_ref.step(json.dumps(action))

        trajectory.append(
            {
                "step": step_idx,
                "obs": obs,
                "action": action,
                "events": events_text.split("\n") if events_text else [],
            }
        )

    return trajectory
