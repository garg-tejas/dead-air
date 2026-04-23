"""Utilities for Unsloth-based GRPO training on Dead Air.

Provides fast generation, episode-level GRPO loss computation,
and reward shaping with a conditional arrival bonus.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


def generate_episode_step(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Tuple[List[str], torch.Tensor, List[torch.Tensor], List[int]]:
    """Generate one action completion per prompt for a batch of envs.

    Uses the model's ``generate()`` for sampling, then does a forward
    pass to compute the exact log-probability of each sampled completion.

    Args:
        model: Unsloth-fast LoRA model.
        tokenizer: Matching tokenizer.
        prompts: List of observation prompt strings.
        max_new_tokens: Hard cap on generation length.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.

    Returns:
        action_texts: Decoded action strings.
        old_log_probs: Total log-prob of each completion [batch_size].
        completion_ids_list: Token-ID tensor for each completion.
        prompt_lens: Token length of each prompt.
    """
    if not prompts:
        return [], torch.tensor([]), [], []

    # Tokenize prompts
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Generate completions (no gradients)
    with torch.no_grad():
        output_sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs.input_ids.shape[1]

    # Compute log-probs of generated tokens via a forward pass
    action_texts = []
    old_log_probs = []
    completion_ids_list = []

    with torch.no_grad():
        # Forward on full sequences: logits shape [batch, seq-1, vocab]
        logits = model(output_sequences).logits[:, :-1, :]
        log_probs_all = F.log_softmax(logits, dim=-1)

        for i in range(len(prompts)):
            comp_ids = output_sequences[i, prompt_len:]
            completion_ids_list.append(comp_ids)

            if comp_ids.numel() == 0:
                action_texts.append("")
                old_log_probs.append(
                    torch.tensor(0.0, device=model.device)
                )
                continue

            # Gather log-prob of each generated token
            token_lps = []
            for j in range(comp_ids.shape[0]):
                tok_id = comp_ids[j]
                # log_probs_all[i, pos] predicts token at pos+1
                lp = log_probs_all[i, prompt_len + j - 1, tok_id]
                token_lps.append(lp)

            old_log_probs.append(torch.stack(token_lps).sum())
            action_texts.append(
                tokenizer.decode(comp_ids, skip_special_tokens=True)
            )

    return (
        action_texts,
        torch.stack(old_log_probs),
        completion_ids_list,
        [prompt_len] * len(prompts),
    )


def compute_grpo_loss(
    model,
    tokenizer,
    episodes_data: List[List[Tuple[str, torch.Tensor, int, torch.Tensor]]],
    rewards: List[float],
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Compute the GRPO clipped surrogate loss for a batch of episodes.

    Each episode is a list of (prompt_text, completion_ids, prompt_len,
    old_log_prob) tuples --- one per step.  All steps in an episode share
    the same advantage (the normalized episode reward).

    Args:
        episodes_data: Per-episode step data.
        rewards: Episode-level scalar rewards [num_episodes].
        epsilon: PPO clipping parameter (default 0.2).

    Returns:
        Scalar loss tensor with ``requires_grad=True``.
    """
    if not episodes_data or not rewards:
        return torch.tensor(
            0.0, device=model.device, requires_grad=True
        )

    # ----- 1.  Per-group advantage normalization -----
    rewards_t = torch.tensor(
        rewards, dtype=torch.float32, device=model.device
    )
    mean_r = rewards_t.mean()
    std_r = rewards_t.std() + 1e-8
    advantages = (rewards_t - mean_r) / std_r          # [num_episodes]

    # ----- 2.  Flatten all steps across all episodes -----
    all_prompts = []
    all_completion_ids = []
    all_prompt_lens = []
    all_old_log_probs = []
    episode_indices = []

    for ep_idx, episode in enumerate(episodes_data):
        for prompt_text, comp_ids, p_len, old_lp in episode:
            all_prompts.append(prompt_text)
            all_completion_ids.append(comp_ids)
            all_prompt_lens.append(p_len)
            all_old_log_probs.append(old_lp)
            episode_indices.append(ep_idx)

    if not all_prompts:
        return torch.tensor(
            0.0, device=model.device, requires_grad=True
        )

    # ----- 3.  Recompute log-probs under *current* policy -----
    # Re-tokenize prompts
    prompt_inputs = tokenizer(
        all_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Pad completion IDs to uniform length for batching
    max_comp_len = max(c.shape[0] for c in all_completion_ids)
    padded_comp = torch.full(
        (len(all_completion_ids), max_comp_len),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=model.device,
    )
    comp_mask = torch.zeros(
        (len(all_completion_ids), max_comp_len),
        dtype=torch.float32,
        device=model.device,
    )

    for i, cids in enumerate(all_completion_ids):
        L = cids.shape[0]
        padded_comp[i, :L] = cids
        comp_mask[i, :L] = 1.0

    # Concatenate prompt + completion
    full_ids = torch.cat([prompt_inputs.input_ids, padded_comp], dim=1)
    full_mask = torch.cat(
        [prompt_inputs.attention_mask, comp_mask], dim=1
    )

    # Forward pass (gradients enabled)
    logits = model(
        input_ids=full_ids, attention_mask=full_mask
    ).logits[:, :-1, :]
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Gather per-step log-probs
    new_log_probs = []
    for i in range(len(all_prompts)):
        p_len = all_prompt_lens[i]
        comp_len = int(comp_mask[i].sum().item())

        if comp_len == 0:
            new_log_probs.append(
                torch.tensor(0.0, device=model.device)
            )
            continue

        token_lps = []
        for j in range(comp_len):
            tok_id = padded_comp[i, j]
            # log_probs_all[i, p_len + j] predicts token p_len + j + 1
            lp = log_probs_all[i, p_len + j, tok_id]
            token_lps.append(lp)

        new_log_probs.append(torch.stack(token_lps).sum())

    new_log_probs = torch.stack(new_log_probs)
    old_log_probs = torch.stack(all_old_log_probs).to(model.device)

    # ----- 4.  Match advantages to every step -----
    step_advantages = advantages[
        torch.tensor(episode_indices, device=model.device)
    ]

    # ----- 5.  Clipped surrogate loss -----
    ratio = torch.exp(new_log_probs - old_log_probs.detach())
    clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
    loss = -torch.min(
        ratio * step_advantages, clipped * step_advantages
    ).mean()

    return loss


def compute_episode_reward(
    env, enable_arrival_bonus: bool = False
) -> float:
    """Return episode reward with an optional arrival bonus.

    The bonus rewards units that successfully reached their assigned
    calls (``on_scene`` or ``returning``).  It is deliberately small
    (+0.03 per unit) and capped so it cannot dominate the survival
    signal.
    """
    base = env.reward if env.reward is not None else 0.0

    if not enable_arrival_bonus:
        return base

    arrival_count = sum(
        1
        for u in env.env.units
        if u.status in ("on_scene", "returning")
    )
    bonus = arrival_count * 0.03
    return min(base + bonus, 1.0)


def should_enable_arrival_bonus(
    reward_history: List[float], min_episodes: int = 20
) -> bool:
    """Decide whether to turn on the arrival bonus.

    Criteria (both must hold):
    1.  We have seen at least ``min_episodes`` episodes.
    2.  The variance of the last ``min_episodes`` rewards is < 0.01
        (i.e. the curve has flattened out).
    """
    if len(reward_history) < min_episodes:
        return False

    recent = reward_history[-min_episodes:]
    return float(np.var(recent)) < 0.01
