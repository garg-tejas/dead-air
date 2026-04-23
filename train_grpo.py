"""GRPO training with TRL + vLLM for Dead Air.

Uses TRL's built-in GRPOTrainer with ``environment_factory`` so the model can
interact with the environment across multiple turns. The prompt and generation
settings target Qwen 3.5 thinking models while keeping the final action easy to
parse as compact JSON.
"""

import argparse
import importlib.util
import os
import sys
import types
from typing import Any, List

from datasets import Dataset


SYSTEM_PROMPT = """You are the emergency dispatch commander for a 20-node city.
You control 6 units and must minimize fatalities, response time, and coverage gaps.

Reason through the situation internally before you answer. Do not reveal your
chain-of-thought. Your visible response must contain only one compact JSON object
on the final line with no markdown fences and no extra commentary.

Valid JSON schemas:
{"action_type":"dispatch","unit_id":0,"call_id":1}
{"action_type":"reroute","unit_id":0,"call_id":1}
{"action_type":"stage","unit_id":0,"location_node":5}
{"action_type":"divert","unit_id":0,"hospital_id":1}
{"action_type":"verify","call_id":1}
{"action_type":"request_mutual_aid"}
{"action_type":"log","note":"short plain text note"}
{"action_type":"hold"}

Rules:
- Return exactly one JSON object.
- Use exactly the key order shown in the schema examples.
- Use integer IDs.
- If the best action is unclear, return {"action_type":"hold"}.
"""

ACTION_JSON_REGEX = (
    r'(\{"action_type":"dispatch","unit_id":[0-9]+,"call_id":[0-9]+\}'
    r'|\{"action_type":"reroute","unit_id":[0-9]+,"call_id":[0-9]+\}'
    r'|\{"action_type":"stage","unit_id":[0-9]+,"location_node":[0-9]+\}'
    r'|\{"action_type":"divert","unit_id":[0-9]+,"hospital_id":[0-9]+\}'
    r'|\{"action_type":"verify","call_id":[0-9]+\}'
    r'|\{"action_type":"request_mutual_aid"\}'
    r'|\{"action_type":"log","note":"[^"\n]{0,120}"\}'
    r'|\{"action_type":"hold"\})'
)


def patch_transformers_cache_compat() -> None:
    """Restore TRANSFORMERS_CACHE for optional dependencies expecting it."""
    try:
        import transformers.utils.hub as hub
    except Exception:
        return

    if not hasattr(hub, "TRANSFORMERS_CACHE"):
        hub.TRANSFORMERS_CACHE = os.environ.get(
            "TRANSFORMERS_CACHE",
            os.path.expanduser("~/.cache/huggingface/transformers"),
        )


def patch_vllm_ascend_compat() -> None:
    """Stub TRL's optional vllm_ascend import on CUDA systems."""
    if importlib.util.find_spec("vllm_ascend") is not None:
        return

    try:
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )
    except Exception:
        class PyNcclCommunicator:  # type: ignore[no-redef]
            pass

    root = types.ModuleType("vllm_ascend")
    distributed = types.ModuleType("vllm_ascend.distributed")
    device_comms = types.ModuleType(
        "vllm_ascend.distributed.device_communicators"
    )
    pyhccl = types.ModuleType(
        "vllm_ascend.distributed.device_communicators.pyhccl"
    )
    pyhccl.PyHcclCommunicator = PyNcclCommunicator

    sys.modules[root.__name__] = root
    sys.modules[distributed.__name__] = distributed
    sys.modules[device_comms.__name__] = device_comms
    sys.modules[pyhccl.__name__] = pyhccl


def make_reward_func() -> Any:
    """Return episode reward from each environment instance."""

    def reward_func(environments: List[Any], **_: Any) -> List[float]:
        rewards = []
        for env in environments:
            rewards.append(float(env.reward) if env.reward is not None else 0.0)
        return rewards

    return reward_func


def build_dataset(episodes: int, difficulty: str) -> Dataset:
    """Create the conversational dataset expected by TRL environment_factory."""
    return Dataset.from_dict(
        {
            "prompt": [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Review the latest city state and return the next "
                            "dispatch action as one JSON object."
                        ),
                    },
                ]
            ]
            * episodes,
            "difficulty": [difficulty] * episodes,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--difficulty", type=str, default="curriculum")
    parser.add_argument("--output-dir", type=str, default="./outputs/grpo")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-vllm", action="store_true")
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate", "server"],
        default="colocate",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization", type=float, default=0.25
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-server-host", type=str, default="0.0.0.0")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-server-base-url", type=str, default=None)
    parser.add_argument("--vllm-server-timeout", type=float, default=600.0)
    parser.add_argument(
        "--vllm-structured-outputs-regex",
        type=str,
        default=ACTION_JSON_REGEX,
    )
    parser.add_argument(
        "--vllm-sleep-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    print("=" * 60)
    print("Dead Air GRPO Training")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    import torch

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"BF16 supported: {bf16}")

    patch_transformers_cache_compat()
    patch_vllm_ascend_compat()

    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise ImportError(
            "TRL is required. Install a Lightning-compatible TRL + vLLM stack first."
        ) from exc

    from peft import LoraConfig

    try:
        from server.grpo_env_wrapper import DeadAirGRPOEnv
    except ImportError:
        from dead_air.server.grpo_env_wrapper import DeadAirGRPOEnv

    dataset = build_dataset(args.episodes, args.difficulty)
    reward_func = make_reward_func()

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=args.save_every,
        report_to="none",
        seed=args.seed,
        bf16=bf16,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        repetition_penalty=args.repetition_penalty,
        generation_kwargs={
            "presence_penalty": args.presence_penalty,
            "enable_thinking": True,
        },
        chat_template_kwargs={"enable_thinking": True},
        use_vllm=not args.no_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_enable_sleep_mode=args.vllm_sleep_mode,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_structured_outputs_regex=(
            args.vllm_structured_outputs_regex if not args.no_vllm else None
        ),
        scale_rewards="batch",
        beta=0.0,
    )

    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_func,
        environment_factory=DeadAirGRPOEnv,
        peft_config=peft_config,
    )

    print("\nStarting training...")
    trainer.train()

    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    trainer.save_model(final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()
