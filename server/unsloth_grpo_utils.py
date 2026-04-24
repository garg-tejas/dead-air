"""Utilities for Unsloth-based GRPO training on Dead Air.

Provides fast generation, episode-level GRPO loss computation,
and reward shaping with a conditional arrival bonus.
"""

import torch
import torch.nn.functional as F
from typing import List


def compute_grpo_loss(
    model,
    tokenizer,
    episodes: List[List[dict]],
    rewards: List[float],
    epsilon: float = 0.2,
    micro_batch_size: int = 1,
) -> float:
    """Compute the GRPO clipped surrogate loss for a batch of episodes.

    Each episode is a list of step dicts:
        { "prompt_ids": torch.LongTensor (CPU),
          "completion_ids": torch.LongTensor (CPU),
          "old_log_prob": torch.Tensor (scalar) }

    All steps in an episode share the same advantage (the normalized
    episode reward).

    Args:
        model: Unsloth-fast LoRA model.
        tokenizer: Matching tokenizer.
        episodes: Per-episode step data.
        rewards: Episode-level scalar rewards [num_episodes].
        epsilon: PPO clipping parameter (default 0.2).
        micro_batch_size: Number of steps per micro-batch during backward.

    Returns:
        Scalar loss value (Python float).
    """
    if not episodes or not rewards or not any(episodes):
        return 0.0

    device = next(model.parameters()).device

    # ----- 1.  Per-group advantage normalization -----
    rewards_t = torch.tensor(
        rewards, dtype=torch.float32, device=device
    )
    mean_r = rewards_t.mean()
    std_r = rewards_t.std() + 1e-8
    advantages = (rewards_t - mean_r) / std_r  # [num_episodes]

    # ----- 2.  Flatten all steps across all episodes -----
    all_prompt_ids = []
    all_completion_ids = []
    all_old_log_probs = []
    episode_indices = []

    for ep_idx, episode in enumerate(episodes):
        for step in episode:
            all_prompt_ids.append(step["prompt_ids"].to(device))
            all_completion_ids.append(step["completion_ids"].to(device))
            all_old_log_probs.append(
                step["old_log_prob"].to(device)
                if step["old_log_prob"].device != device
                else step["old_log_prob"]
            )
            episode_indices.append(ep_idx)

    total_steps = len(all_prompt_ids)
    if total_steps == 0:
        return 0.0

    # ----- 3.  Micro-batched recompute of log-probs -----
    total_loss = 0.0
    total_count = 0

    for mb_start in range(0, total_steps, micro_batch_size):
        mb_end = min(mb_start + micro_batch_size, total_steps)
        mb_indices = range(mb_start, mb_end)

        # Pad prompt + completion to same length for batching
        full_ids_list = [
            torch.cat([
                all_prompt_ids[i],
                all_completion_ids[i],
            ])
            for i in mb_indices
        ]
        max_len = max(t.shape[0] for t in full_ids_list)

        full_ids_batch = torch.full(
            (len(mb_indices), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(
            (len(mb_indices), max_len),
            dtype=torch.long,
            device=device,
        )
        for i_idx, i in enumerate(mb_indices):
            ids = full_ids_list[i_idx]
            full_ids_batch[i_idx, : ids.shape[0]] = ids
            attention_mask[i_idx, : ids.shape[0]] = 1

        # Forward pass (gradients enabled)
        logits = model(
            input_ids=full_ids_batch,
            attention_mask=attention_mask,
        ).logits[:, :-1, :]
        log_probs_all = F.log_softmax(logits, dim=-1)

        # Gather per-step log-probs
        mb_new_log_probs = []
        for i_idx, i in enumerate(mb_indices):
            p_len = all_prompt_ids[i].shape[0]
            c_len = all_completion_ids[i].shape[0]
            if c_len == 0:
                mb_new_log_probs.append(
                    torch.tensor(0.0, device=device)
                )
                continue

            token_lps = []
            for j in range(c_len):
                tok_id = full_ids_batch[i_idx, p_len + j]
                lp = log_probs_all[i_idx, p_len + j - 1, tok_id]
                token_lps.append(lp)
            mb_new_log_probs.append(torch.stack(token_lps).sum())

        mb_new_log_probs = torch.stack(mb_new_log_probs)
        mb_old_log_probs = torch.stack([
            all_old_log_probs[i] for i in mb_indices
        ])
        mb_advantages = advantages[
            torch.tensor(
                [episode_indices[i] for i in mb_indices],
                device=device,
            )
        ]

        # Clipped surrogate loss
        ratio = torch.exp(mb_new_log_probs - mb_old_log_probs.detach())
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        loss = -torch.min(
            ratio * mb_advantages,
            clipped * mb_advantages,
        ).mean()

        loss.backward()
        total_loss += loss.item() * len(mb_indices)
        total_count += len(mb_indices)

    return total_loss / total_count
