"""Export a single DispatchR episode to JSON for visualization.

Usage:
    python export_episode.py --difficulty learning --output episode_001.json

Runs a greedy-agent episode and captures full state at every step.
"""

import argparse
import json
import sys

sys.path.insert(0, ".")

from server.dispatcher_environment import DispatcherEnvironment


def _dedup_calls(calls):
    """Deduplicate calls by call_id (a call may appear in both active and resolved lists)."""
    seen = set()
    result = []
    for c in calls:
        cid = c["call_id"]
        if cid not in seen:
            seen.add(cid)
            result.append(c)
    return result


def run_and_export(difficulty: str = "learning", output: str = "episode.json", seed: int = 42):
    env = DispatcherEnvironment(seed=seed)
    # Disable radio delay for deterministic, consistent episode export
    env.radio_buffer.delay_prob = 0.0
    obs = env.reset(difficulty=difficulty)

    episode_log = []
    done = False
    step = 0

    while not done and step < 80:
        # Build greedy action
        action = greedy_action(env)
        obs = env.step(action)
        step += 1
        done = obs.get("done", False)

        # Capture full state
        state = {
            "step": step,
            "max_steps": 80,
            "action": action,
            "reward": obs.get("reward"),
            "recent_events": obs.get("recent_events", []),
            "traffic_alerts": obs.get("traffic_alerts", []),
            "hospital_statuses": [
                {"hospital_id": h["hospital_id"], "reported_status": h["reported_status"]}
                for h in obs.get("hospital_statuses", [])
            ],
            "mutual_aid_remaining": obs.get("mutual_aid_remaining", 0),
            "units": [
                {
                    "unit_id": u.unit_id,
                    "location": u.location,
                    "status": u.status,
                    "current_call": u.current_call,
                    "current_call_type": u.current_call_type,
                    "path_remaining": u.path_remaining.copy(),
                    "reliability": u.reliability,
                }
                for u in env.units
            ],
            "calls": [
                {
                    "call_id": c["call_id"],
                    "location": c["location"],
                    "call_type": c["call_type"],
                    "reported_type": c["reported_type"],
                    "caller_tone": c["caller_tone"],
                    "time_elapsed": c["time_elapsed"],
                    "time_received": c["time_received"],
                    "assigned_unit": c.get("assigned_unit"),
                    "resolved": c["resolved"],
                    "fatality": c.get("fatality", False),
                    "is_ghost": c.get("is_ghost", False),
                    "is_false_alarm": c.get("is_false_alarm", False),
                    "severity_modifier": c.get("severity_modifier", 1.0),
                    "panic_modifier": c.get("panic_modifier", 1.0),
                }
                for c in _dedup_calls(env.call_generator.active_calls + env.call_generator.resolved_calls)
            ],
        }
        episode_log.append(state)

    with open(output, "w") as f:
        json.dump(episode_log, f, indent=2)

    print(f"Exported {len(episode_log)} steps to {output}")
    final_reward = episode_log[-1]["reward"] if episode_log else None
    print(f"Final reward: {final_reward}")


def greedy_action(env):
    """Greedy dispatch: closest idle unit to highest-priority active call."""
    active = [c for c in env.call_generator.get_active() if not c["resolved"]]
    if not active:
        return {"action_type": "hold"}

    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
    sorted_calls = sorted(
        active,
        key=lambda c: priority.get(c.get("reported_type", "trauma"), 1),
        reverse=True,
    )

    for call in sorted_calls:
        best_unit = None
        best_dist = float("inf")
        for u in env.units:
            if u.status == "idle":
                dist = env.city_graph.travel_time(u.location, call["location"])
                if dist < best_dist:
                    best_dist = dist
                    best_unit = u.unit_id
        if best_unit is not None:
            return {
                "action_type": "dispatch",
                "unit_id": best_unit,
                "call_id": call["call_id"],
            }
    return {"action_type": "hold"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", default="learning")
    parser.add_argument("--output", default="episode.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_and_export(args.difficulty, args.output, args.seed)
