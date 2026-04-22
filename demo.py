"""Interactive demo script for Dead Air environment.

Run:
    uv run python demo.py --episodes 3 --difficulty learning
"""

import argparse
import time

from dead_air.server.dispatcher_environment import DispatcherEnvironment


def print_observation(obs: dict, step: int):
    """Pretty-print observation in terminal."""
    print(f"\n{'='*60}")
    print(f"STEP {step:2d} | Shift: {obs['step_number']}/{obs['max_steps']}")
    print(f"{'='*60}")

    # Units
    print("\n[UNITS]:")
    for u in obs.get("unit_statuses", []):
        call_info = f" -> Call {u['current_call']}" if u.get("current_call") else ""
        print(f"   Unit {u['unit_id']}: {u['last_known_status']:12s} @ Node {u['last_known_location']:2d}{call_info}")

    # Calls
    calls = obs.get("active_calls", [])
    if calls:
        print("\n[CALLS]:")
        for c in calls:
            assigned = f" (Unit {c['assigned_unit']})" if c.get("assigned_unit") else ""
            print(f"   Call {c['call_id']}: {c['reported_type']:8s} @ Node {c['location']:2d} | "
                  f"Tone: {c['caller_tone']:8s} | Elapsed: {c['time_elapsed']:2d}min{assigned}")
    else:
        print("\n[CALLS]: No active calls")

    # Events
    events = obs.get("recent_events", [])
    if events:
        print("\n[EVENTS]:")
        for e in events[-3:]:
            print(f"   * {e}")

    # Mutual aid
    print(f"\n[Mutual Aid Remaining]: {obs['mutual_aid_remaining']}")

    # Reward at episode end
    if obs.get("reward") is not None:
        print(f"\n[EPISODE REWARD]: {obs['reward']:.3f}")


def greedy_action(obs: dict) -> dict:
    """Greedy dispatcher for demo."""
    calls = obs.get("active_calls", [])
    units = obs.get("unit_statuses", [])

    if not calls:
        return {"action_type": "hold"}

    # Priority
    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
    sorted_calls = sorted(calls, key=lambda c: priority.get(c.get("reported_type"), 1), reverse=True)
    target = sorted_calls[0]

    # Closest idle unit
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


def run_demo(env: DispatcherEnvironment, num_episodes: int = 3, difficulty: str = "learning", delay: float = 0.5):
    """Run interactive demo episodes."""
    for ep in range(num_episodes):
        print(f"\n{'#'*60}")
        print(f"# EPISODE {ep + 1}")
        print(f"{'#'*60}")

        obs = env.reset(difficulty=difficulty)
        # Disable radio delay for demo so greedy agent sees true status
        env.radio_buffer.delay_prob = 0.0
        # Also seed last_known_statuses with fresh true statuses
        for u in env.units:
            env._last_known_statuses[u.unit_id] = u.get_observable_status()
        done = False
        step = 0

        while not done and step < 85:
            print_observation(obs, step)
            action = greedy_action(obs)
            print(f"\n[AGENT] Action: {action}")

            obs = env.step(action)
            done = obs.get("done", False)
            step += 1

            if delay > 0:
                time.sleep(delay)

        print(f"\n{'#'*60}")
        print(f"# EPISODE {ep + 1} COMPLETE")
        print(f"# Reward: {obs.get('reward', 0.0):.3f}")
        print(f"{'#'*60}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--difficulty", type=str, default="learning")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between steps (0 for instant)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = DispatcherEnvironment(seed=args.seed)
    run_demo(env, num_episodes=args.episodes, difficulty=args.difficulty, delay=args.delay)


if __name__ == "__main__":
    main()
