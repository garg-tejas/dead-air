"""Diagnostic script: run episodes with detailed action/reward logging.

Usage:
    uv run python diagnose.py --episodes 10 --agent greedy
    uv run python diagnose.py --episodes 5 --agent random
"""

import argparse
import json
from typing import Any, Dict, List

from dead_air.server.dispatcher_environment import DispatcherEnvironment


def greedy_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy dispatcher."""
    calls = obs.get("active_calls", [])
    units = obs.get("unit_statuses", [])
    if not calls:
        return {"action_type": "hold"}
    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
    sorted_calls = sorted(calls, key=lambda c: priority.get(c.get("reported_type"), 1), reverse=True)
    target = sorted_calls[0]
    best_unit = None
    best_dist = float("inf")
    for u in units:
        if u.get("last_known_status") == "idle":
            dist = abs(u.get("last_known_location", 0) - target["location"])
            if dist < best_dist:
                best_dist = dist
                best_unit = u["unit_id"]
    if best_unit is not None:
        return {"action_type": "dispatch", "unit_id": best_unit, "call_id": target["call_id"]}
    return {"action_type": "hold"}


def random_action(obs: Dict[str, Any], rng) -> Dict[str, Any]:
    """Random valid action."""
    import numpy as np
    calls = obs.get("active_calls", [])
    units = obs.get("unit_statuses", [])
    idle_units = [u for u in units if u.get("last_known_status") == "idle"]
    choices = [{"action_type": "hold"}]
    if calls and idle_units:
        for c in calls:
            for u in idle_units:
                choices.append({"action_type": "dispatch", "unit_id": u["unit_id"], "call_id": c["call_id"]})
    if rng.random() < 0.05 and calls:
        choices.append({"action_type": "verify", "call_id": calls[0]["call_id"]})
    return choices[rng.integers(0, len(choices))]


def run_diagnostic(env: DispatcherEnvironment, num_episodes: int = 10, agent_type: str = "greedy", seed: int = 42) -> List[Dict]:
    import numpy as np
    rng = np.random.default_rng(seed)
    logs = []

    for ep in range(num_episodes):
        obs = env.reset(difficulty="learning")
        # Disable radio delay for greedy baseline so it sees true status
        if agent_type == "greedy":
            env.radio_buffer.delay_prob = 0.0
        done = False
        step = 0
        episode_actions = []
        episode_events = []

        while not done and step < 85:
            if agent_type == "greedy":
                action = greedy_action(obs)
            else:
                action = random_action(obs, rng)

            episode_actions.append(action)
            obs = env.step(action)
            done = obs.get("done", False)
            episode_events.extend(obs.get("recent_events", []))
            step += 1

        reward = obs.get("reward", 0.0) or 0.0
        gt = env.get_ground_truth()
        logs.append({
            "episode": ep + 1,
            "steps": step,
            "reward": reward,
            "fatalities": gt["fatality_count"],
            "calls_total": len(gt["calls"]),
            "calls_resolved": sum(1 for c in gt["calls"] if c["resolved"]),
            "action_histogram": _histogram(episode_actions),
            "event_histogram": _histogram_events(episode_events),
            "last_10_events": episode_events[-10:],
        })

    return logs


def _histogram(actions: List[Dict]) -> Dict[str, int]:
    hist = {}
    for a in actions:
        t = a.get("action_type", "unknown")
        hist[t] = hist.get(t, 0) + 1
    return hist


def _histogram_events(events: List[str]) -> Dict[str, int]:
    hist = {}
    for e in events:
        if "delayed" in e.lower():
            hist["unit_delay"] = hist.get("unit_delay", 0) + 1
        elif "cleared call" in e.lower():
            hist["call_cleared"] = hist.get("call_cleared", 0) + 1
        elif "new call" in e.lower():
            hist["new_call"] = hist.get("new_call", 0) + 1
        elif "mutual aid" in e.lower():
            hist["mutual_aid"] = hist.get("mutual_aid", 0) + 1
        elif "verified" in e.lower():
            hist["verify"] = hist.get("verify", 0) + 1
        elif "invalid" in e.lower():
            hist["invalid_action"] = hist.get("invalid_action", 0) + 1
    return hist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--agent", type=str, default="greedy", choices=["greedy", "random"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="diagnose.json")
    args = parser.parse_args()

    env = DispatcherEnvironment(seed=args.seed)
    logs = run_diagnostic(env, num_episodes=args.episodes, agent_type=args.agent, seed=args.seed)

    rewards = [l["reward"] for l in logs]
    fatalities = [l["fatalities"] for l in logs]
    print("=" * 60)
    print(f"DIAGNOSTIC: {args.agent} agent, {args.episodes} episodes")
    print("=" * 60)
    print(f"Mean reward:    {sum(rewards)/len(rewards):.3f}")
    print(f"Reward range:   {min(rewards):.3f} - {max(rewards):.3f}")
    print(f"Mean fatalities:{sum(fatalities)/len(fatalities):.2f}")
    print(f"Max fatalities: {max(fatalities)}")
    print()

    for log in logs:
        print(f"Ep {log['episode']:2d} | Reward: {log['reward']:.3f} | Steps: {log['steps']:2d} | "
              f"Calls: {log['calls_resolved']}/{log['calls_total']} | Fatalities: {log['fatalities']}")
        print(f"       Actions: {log['action_histogram']}")
        if log['event_histogram']:
            print(f"       Events:  {log['event_histogram']}")
        print()

    with open(args.output, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"Saved full logs to {args.output}")


if __name__ == "__main__":
    main()
