"""Evaluation script: greedy baseline vs oracle comparison."""

from typing import Any, Dict, List, Set

try:
    from dead_air.server.dispatcher_environment import DispatcherEnvironment
except ImportError:
    from server.dispatcher_environment import DispatcherEnvironment


def greedy_agent_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy dispatcher: send closest idle unit to highest-priority active call."""
    active_calls = obs.get("active_calls", [])
    unit_statuses = obs.get("unit_statuses", [])

    if not active_calls:
        return {"action_type": "hold"}

    # Prioritize by reported type severity
    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
    sorted_calls = sorted(
        active_calls,
        key=lambda c: priority.get(c.get("reported_type", "trauma"), 1),
        reverse=True,
    )

    for call in sorted_calls:
        call_id = call["call_id"]
        call_loc = call["location"]

        # Find closest idle unit using node distance (proxy for graph distance)
        best_unit = None
        best_dist = float("inf")
        for u in unit_statuses:
            if u.get("last_known_status") == "idle":
                dist = abs(u.get("last_known_location", 0) - call_loc)
                if dist < best_dist:
                    best_dist = dist
                    best_unit = u["unit_id"]

        if best_unit is not None:
            return {"action_type": "dispatch", "unit_id": best_unit, "call_id": call_id}

    return {"action_type": "hold"}


def run_episodes(
    env: DispatcherEnvironment,
    num_episodes: int = 10,
    agent_type: str = "greedy",
    difficulty: str = "learning",
) -> List[float]:
    """Run episodes and return rewards."""
    rewards = []
    for ep in range(num_episodes):
        obs = env.reset(difficulty=difficulty)
        # Disable radio delay for baseline evaluation so greedy agent sees true status
        env.radio_buffer.delay_prob = 0.0
        # Also seed last_known_statuses with fresh true statuses
        for u in env.units:
            env._last_known_statuses[u.unit_id] = u.get_observable_status()
        done = False
        step_count = 0
        while not done and step_count < 100:
            if agent_type == "greedy":
                action = greedy_agent_step(obs)
            else:
                action = {"action_type": "hold"}
            obs = env.step(action)
            done = obs.get("done", False)
            step_count += 1
        # Reward is set by env.step() at episode end
        reward = obs.get("reward", 0.0) or 0.0
        rewards.append(reward)
    return rewards


def main():
    print("Running Greedy Baseline Evaluation...")
    env = DispatcherEnvironment(seed=42)
    rewards = run_episodes(env, num_episodes=10, agent_type="greedy", difficulty="learning")
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    print(f"Mean reward over {len(rewards)} episodes: {mean_reward:.3f}")
    print(f"Rewards: {[round(r, 3) for r in rewards]}")


if __name__ == "__main__":
    main()
