"""Trajectory cache system for DispatchR.

Builds, persists, and samples from pre-computed greedy episode trajectories.
This eliminates the need to re-run episode simulation during dataset construction.

Cache format (JSONL, one line per episode):
    {
        "seed": 123,
        "difficulty": "learning",
        "episode_reward": 0.45,
        "steps": [
            {"step": 0, "action": {...}},
            {"step": 1, "action": {...}},
            ...
        ]
    }

We do NOT store full observations in the cache to keep file size reasonable.
Observations are cheaply recomputed by replaying cached actions.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .greedy_policy import greedy_action
from .grpo_env_wrapper import DispatchRGRPOEnv


def _run_single_episode(args: Tuple[int, str]) -> Dict[str, Any]:
    """Worker: run one greedy episode. Must be top-level for pickling."""
    seed, difficulty = args
    env = DispatchRGRPOEnv(seed=seed, difficulty=difficulty)
    env.reset(seed=seed, difficulty=difficulty)

    steps = []
    for step_idx in range(80):
        obs = env._obs
        if obs is None or obs.get("done"):
            break
        action = greedy_action(obs)
        env.step(action)
        steps.append({"step": step_idx, "action": action})

    reward = env.reward if env.reward is not None else 0.0
    return {
        "seed": seed,
        "difficulty": difficulty,
        "episode_reward": float(reward),
        "steps": steps,
    }


def build_trajectory_cache(
    n_seeds: int,
    difficulty: str,
    output_path: str,
    base_seed: int = 0,
    num_workers: int = 8,
) -> str:
    """Build trajectory cache by running greedy episodes in parallel.

    Args:
        n_seeds: Number of episode seeds to simulate.
        difficulty: Curriculum phase string.
        output_path: Where to write the JSONL cache file.
        base_seed: Offset for seed values.
        num_workers: Parallel workers (default 8, set to 1 for debugging).

    Returns:
        Path to the created cache file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    tasks = [(base_seed + i, difficulty) for i in range(n_seeds)]

    print(f"[Cache] Building {n_seeds} greedy trajectories (difficulty={difficulty})...")
    print(f"[Cache] Workers: {num_workers}")

    with open(output_path, "w", encoding="utf-8") as f:
        if num_workers <= 1:
            # Sequential (useful for debugging)
            for task in tasks:
                episode = _run_single_episode(task)
                f.write(json.dumps(episode) + "\n")
                f.flush()
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_run_single_episode, t): t for t in tasks}
                for future in as_completed(futures):
                    episode = future.result()
                    f.write(json.dumps(episode) + "\n")
                    f.flush()

    print(f"[Cache] Saved to {output_path}")
    return output_path


def load_trajectory_cache(cache_path: str) -> List[Dict[str, Any]]:
    """Load full cache into memory.

    Returns list of episode dicts. For large caches consider streaming
    with ``iter_trajectory_cache`` instead.
    """
    episodes = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    return episodes


def iter_trajectory_cache(cache_path: str):
    """Stream trajectory cache line-by-line (memory-efficient)."""
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sample_dataset_rows(
    cache_path: str,
    tokenizer,
    system_prompt: str,
    n_samples_per_seed: int = 5,
    max_step: int = 79,
    build_chat_prompt_fn=None,
) -> List[Dict[str, Any]]:
    """Sample random-step observations from cache to build dataset rows.

    For each episode in the cache, sample ``n_samples_per_seed`` random steps
    (uniformly), replay the episode to that step, capture the observation,
    and format it as a prompt.

    Args:
        cache_path: Path to JSONL cache file.
        tokenizer: HF tokenizer for chat template formatting.
        system_prompt: The SYSTEM_PROMPT string.
        n_samples_per_seed: How many steps to sample per episode.
        max_step: Maximum step index to sample (inclusive).
        build_chat_prompt_fn: Callable(tokenizer, system, user) -> str.

    Returns:
        List of dataset row dicts with keys: seed, difficulty, step_idx, prompt.
    """
    rows = []

    for episode in iter_trajectory_cache(cache_path):
        seed = episode["seed"]
        difficulty = episode["difficulty"]
        cached_steps = episode["steps"]
        n_steps = len(cached_steps)

        if n_steps <= 1:
            continue

        # Sample step indices (exclude step 0, cap at actual episode length)
        upper = min(max_step, n_steps - 1)
        if upper < 1:
            continue

        sampled_indices = np.random.choice(
            range(1, upper + 1),
            size=min(n_samples_per_seed, upper),
            replace=False,
        ).tolist()

        # Replay episode and capture observations at sampled steps
        env = DispatchRGRPOEnv(seed=seed, difficulty=difficulty)
        env.reset(seed=seed, difficulty=difficulty)

        for step_idx in range(n_steps):
            obs = env._obs
            if obs is None or obs.get("done"):
                break

            if step_idx in sampled_indices:
                # Format observation as prompt
                from .prompt_utils import format_observation, build_chat_prompt

                obs_text = format_observation(obs)
                if build_chat_prompt_fn is not None:
                    prompt = build_chat_prompt_fn(tokenizer, system_prompt, obs_text)
                else:
                    prompt = build_chat_prompt(tokenizer, system_prompt, obs_text)

                rows.append(
                    {
                        "seed": seed,
                        "difficulty": difficulty,
                        "step_idx": step_idx,
                        "prompt": prompt,
                    }
                )

            # Apply cached action to advance env
            action = cached_steps[step_idx]["action"]
            env.step(action)

    return rows


def replay_steps(env_ref, cached_steps: List[Dict[str, Any]], up_to_step: int) -> None:
    """Replay cached actions on an env up to (but not including) a target step.

    Used in reward_fn to quickly reconstruct the state at step ``up_to_step``.
    """
    for step_idx in range(min(up_to_step, len(cached_steps))):
        action = cached_steps[step_idx]["action"]
        env_ref.step(action)


def get_cache_path(cache_dir: str, difficulty: str, n_seeds: int) -> str:
    """Return standard cache file path."""
    return os.path.join(
        cache_dir, f"trajectories_{difficulty}_{n_seeds}seeds.jsonl"
    )


def ensure_cache_exists(
    cache_dir: str,
    n_seeds: int,
    difficulty: str,
    base_seed: int = 0,
    num_workers: int = 8,
) -> str:
    """Return cache path, building it first if it doesn't exist."""
    path = get_cache_path(cache_dir, difficulty, n_seeds)
    if os.path.exists(path):
        print(f"[Cache] Reusing existing cache: {path}")
        return path
    return build_trajectory_cache(
        n_seeds=n_seeds,
        difficulty=difficulty,
        output_path=path,
        base_seed=base_seed,
        num_workers=num_workers,
    )
