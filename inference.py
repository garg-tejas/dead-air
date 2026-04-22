"""Run inference with a trained checkpoint."""

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dead_air.server.dispatcher_environment import DispatcherEnvironment
from train import collect_rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="inference_results.json")
    parser.add_argument("--difficulty", type=str, default="expert")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.device == "cuda" else None,
        trust_remote_code=True,
    )

    env = DispatcherEnvironment(seed=42)
    results = []

    for ep in range(args.episodes):
        rollout = collect_rollout(env, model, tokenizer, device=args.device)
        results.append({
            "episode": ep + 1,
            "reward": rollout["reward"],
            "steps": rollout["steps"],
        })
        print(f"Episode {ep+1}: reward={rollout['reward']:.3f}, steps={rollout['steps']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
