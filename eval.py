"""Evaluation script: greedy baseline vs oracle comparison."""

from typing import Any, Dict, List

from dead_air.server.dispatcher_environment import DispatcherEnvironment


def greedy_agent_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Greedy dispatcher: send closest idle unit to highest-priority active call."""
    active_calls = obs.get("active_calls", [])
    unit_statuses = obs.get("unit_statuses", [])

    if not active_calls:
        return {"action_type": "hold"}

    # Prioritize by reported type severity (simplified)
    priority = {"cardiac": 3, "trauma": 2, "fire": 1, "false_alarm": 0}
    sorted_calls = sorted(
        active_calls,
        key=lambda c: priority.get(c.get("reported_type", "trauma"), 1),
        reverse=True,
    )

    for call in sorted_calls:
        call_id = call["call_id"]
        call_loc = call["location"]

        # Find closest idle unit
        best_unit = None
        best_dist = float("inf")
        for u in unit_statuses:
            if u.get("last_known_status") == "idle":
                # Use unit location as proxy for distance
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
) -> List[float]:
    """Run episodes and return rewards."""
    rewards = []
    for ep in range(num_episodes):
        obs = env.reset(difficulty="learning")
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
        # Compute reward at end
        from dead_air.server.reward import RewardComputer
        rc = RewardComputer(env.city_graph)
        gt = env.get_ground_truth()
        result = rc.compute_episode_reward(
            calls=gt["calls"],
            units=env.units,
            oracle_assignments=gt["optimal_assignments"],
        )
        rewards.append(result["episode_reward"])
    return rewards


def main():
    print("Running Greedy Baseline Evaluation...")
    env = DispatcherEnvironment(seed=42)
    rewards = run_episodes(env, num_episodes=10, agent_type="greedy")
    print(f"Mean reward over 10 episodes: {sum(rewards)/len(rewards):.3f}")
    print(f"Rewards: {[round(r, 3) for r in rewards]}")


if __name__ == "__main__":
    main()
