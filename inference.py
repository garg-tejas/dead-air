"""Run inference with a trained DispatchR checkpoint.

Uses the same prompt format and stepping logic as train_grpo.py.

Usage:
    python inference.py --model-path ./outputs/grpo/final --episodes 10 --difficulty expert
    python inference.py --model-path Qwen/Qwen3.5-2B --episodes 5 --difficulty learning

    # Before/after comparison (save trajectory with obs for city animation):
    python inference.py --model-path unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \\
        --use-unsloth --episodes 3 --trajectory-file before.jsonl
    python inference.py --model-path ./outputs/final \\
        --use-unsloth --episodes 3 --trajectory-file after.jsonl
"""

import argparse
import copy
import json
import os
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from server.grpo_env_wrapper import DispatchRGRPOEnv

SYSTEM_PROMPT = (
    "You are an emergency dispatch AI managing 6 ambulance units in a 20-node city. "
    "Every step, output exactly one JSON object on the VERY LAST LINE.\n\n"
    "RULES:\n"
    "- If Active Calls is empty or says '(none)', output: {\"action_type\":\"hold\"}\n"
    "- If there are active calls, dispatch the closest idle unit to the most urgent call.\n"
    "- Keep reasoning to 1-2 sentences. Do not overthink.\n"
    "- The JSON must be the very last thing you output. No markdown, no extra text after it.\n\n"
    "ACTIONS:\n"
    '{"action_type":"dispatch","unit_id":0,"call_id":1}\n'
    '{"action_type":"hold"}\n'
    '{"action_type":"verify","call_id":1}\n\n'
    "Example (no calls): All units idle, no active calls. {\"action_type\":\"hold\"}\n"
    "Example (with calls): Call 2 is cardiac (most urgent). Unit 1 is idle and closest. {\"action_type\":\"dispatch\",\"unit_id\":1,\"call_id\":2}"
)


def format_observation(obs: Dict) -> str:
    """Format env observation into prompt text (same as train_grpo.py)."""
    lines = [
        "# Emergency Dispatch Commander",
        f"Step {obs['step_number']}/{obs['max_steps']}",
        "",
        "## Units",
    ]
    for u in obs.get("unit_statuses", []):
        call_info = f" -> Call {u['current_call']}" if u.get("current_call") else ""
        lines.append(
            f"- Unit {u['unit_id']}: {u['last_known_status']} at Node {u['last_known_location']}{call_info}"
        )
    lines.append("")
    lines.append("## Active Calls")
    active_calls = obs.get("active_calls", [])
    if active_calls:
        for c in active_calls:
            assigned = f" (Unit {c['assigned_unit']})" if c.get("assigned_unit") else ""
            lines.append(
                f"- Call {c['call_id']}: {c['reported_type']} at Node {c['location']} ({c['caller_tone']}) elapsed={c['time_elapsed']}min{assigned}"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Traffic & Hospitals")
    for alert in obs.get("traffic_alerts", []):
        lines.append(f"- {alert}")
    for h in obs.get("hospital_statuses", []):
        lines.append(f"- Hospital {h['hospital_id']}: {h['reported_status']}")
    lines.append("")
    lines.append(f"Mutual aid remaining: {obs['mutual_aid_remaining']}")
    lines.append("")
    lines.append("Choose your next action.")
    return "\n".join(lines)


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    """Build a chat-formatted prompt using the tokenizer's chat template.

    Enables thinking mode for Qwen 3/3.5 models.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if tokenizer.chat_template is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": True},
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    return f"{system}\n\n{user}\n\nAssistant:"


def generate_action(
    model, tokenizer, prompt: str, max_new_tokens: int, device: str
) -> str:
    """Generate one completion from a prompt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return completion


def run_episode(
    env: DispatchRGRPOEnv,
    model,
    tokenizer,
    max_steps: int,
    max_new_tokens: int,
    device: str,
    episode_id: int = 0,
) -> Dict:
    """Run one episode and collect detailed results."""
    env.reset()
    steps_data = []
    for step_idx in range(max_steps):
        if env._obs and env._obs.get("done"):
            break

        # Snapshot the observation before stepping (used for city animation)
        obs_snapshot = copy.deepcopy(env._obs) if env._obs else {}

        obs_text = format_observation(env._obs)
        prompt = build_chat_prompt(tokenizer, SYSTEM_PROMPT, obs_text)
        completion = generate_action(model, tokenizer, prompt, max_new_tokens, device)

        # Step the environment (wrapper parses completion internally)
        result = env.step(completion)

        # Parse what action was actually taken
        parsed = env._parse_action(completion)

        steps_data.append(
            {
                "step": step_idx,
                "prompt": prompt,
                "completion": completion,
                "parsed_action": parsed,
                "events": result,
                "obs": obs_snapshot,
            }
        )

    reward = env.reward if env.reward is not None else 0.0
    metrics = env.metrics

    return {
        "episode": episode_id,
        "reward": reward,
        "steps": len(steps_data),
        "metrics": metrics,
        "trajectory": steps_data,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained DispatchR checkpoint"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained checkpoint or HF model ID"
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument("--output", type=str, default="inference_results.json", help="Output JSON path")
    parser.add_argument(
        "--difficulty", type=str, default="expert", help="Difficulty phase: warmup, learning, advanced, expert"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max tokens per generation"
    )
    parser.add_argument(
        "--max-steps", type=int, default=80, help="Max steps per episode"
    )
    parser.add_argument(
        "--trajectory-file", type=str, default=None,
        help="JSONL file to append trajectory steps with obs data (for city animation / before-after comparison).",
    )
    parser.add_argument(
        "--use-unsloth", action="store_true",
        help="Load model via Unsloth FastLanguageModel (faster, 4-bit support).",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load model in 4-bit quantization (only used with --use-unsloth).",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    if args.use_unsloth:
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_path,
                load_in_4bit=args.load_in_4bit,
                max_seq_length=4096,
                dtype=torch.bfloat16,
            )
            FastLanguageModel.for_inference(model)
        except ImportError as exc:
            raise SystemExit("unsloth not installed. Run without --use-unsloth or: pip install unsloth") from exc
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto" if args.device == "cuda" else None,
            trust_remote_code=True,
        )
        if args.device == "cpu":
            model = model.to("cpu")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print(f"Model loaded. Running {args.episodes} episodes at difficulty '{args.difficulty}'...")
    results = []

    for ep in range(args.episodes):
        env = DispatchRGRPOEnv(seed=42 + ep, difficulty=args.difficulty)
        result = run_episode(
            env,
            model,
            tokenizer,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            device=next(model.parameters()).device,
            episode_id=ep + 1,
        )
        results.append(result)
        metrics = result.get("metrics", {}) or {}
        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"reward={result['reward']:.3f}, "
            f"steps={result['steps']}, "
            f"valid={metrics.get('valid_action_rate', 0):.2f}, "
            f"disp={metrics.get('dispatch_rate', 0):.2f}, "
            f"fatalities={metrics.get('fatality_count', 0)}"
        )

    # Save results (full JSON)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {args.output}")

    # Save JSONL trajectory (for city animation / before-after comparison)
    if args.trajectory_file:
        os.makedirs(os.path.dirname(args.trajectory_file) or ".", exist_ok=True)
        with open(args.trajectory_file, "a", encoding="utf-8") as tf:
            for result in results:
                ep = result["episode"]
                reward = result["reward"]
                for step in result["trajectory"]:
                    tf.write(
                        json.dumps(
                            {
                                "episode": ep,
                                "step": step["step"],
                                "reward": reward,
                                "completion": step["completion"],
                                "obs": step.get("obs", {}),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        print(f"Saved trajectory to {args.trajectory_file}")

    # Summary stats
    rewards = [r["reward"] for r in results]
    print(f"\nSummary: mean_reward={sum(rewards)/len(rewards):.3f}, min={min(rewards):.3f}, max={max(rewards):.3f}")


if __name__ == "__main__":
    main()
