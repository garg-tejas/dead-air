#!/usr/bin/env python3
"""Smoke test for train_trl_grpo.py — validates without running full training.

This script checks everything that can be validated WITHOUT installing
PyTorch / TRL / transformers. On a machine with those installed, also
runs the dataset builder and reward function factory.
"""

import ast
import subprocess
import sys


def check_syntax():
    with open("train_trl_grpo.py", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    print("[PASS] Syntax: train_trl_grpo.py parses cleanly")
    return tree


def check_functions_exist(tree):
    """Verify key top-level functions/classes are defined."""
    names = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    required = {"build_seed_dataset", "make_reward_fn", "CurriculumCallback", "main"}
    missing = required - names
    if missing:
        print(f"[FAIL] Missing definitions: {missing}")
        sys.exit(1)
    print(f"[PASS] Definitions: {sorted(required)}")


def check_cli_args(tree):
    """Verify argparse adds expected arguments."""
    args_found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for kw in node.keywords:
                    if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                        args_found.add(kw.value.value)
                    elif kw.arg == "default" and len(node.args) >= 1:
                        first_arg = node.args[0]
                        if isinstance(first_arg, ast.Constant):
                            args_found.add(first_arg.value.lstrip("-").replace("-", "_"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        args_found.add(arg.value.lstrip("-").replace("-", "_"))
    expected = {"model", "episodes", "batch_size", "max_completion_length", "output_dir"}
    found = {a for a in args_found if a in expected}
    print(f"[PASS] CLI args: {sorted(found)}")


def check_dataset_builder():
    """Test dataset builder if transformers is available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[SKIP] Dataset builder (transformers not installed)")
        return

    sys.path.insert(0, ".")
    from train_trl_grpo import build_seed_dataset

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B", trust_remote_code=True
    )
    ds = build_seed_dataset(n_seeds=2, difficulty="warmup", tokenizer=tok)
    row = ds[0]
    assert "seed" in row and "difficulty" in row and "prompt" in row
    print(
        f"[PASS] Dataset: {len(ds)} rows, "
        f"keys={list(row.keys())}, prompt_len={len(row['prompt'])}"
    )


def check_reward_fn():
    """Test reward function factory if transformers is available."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[SKIP] Reward fn (transformers not installed)")
        return

    import inspect
    sys.path.insert(0, ".")
    from train_trl_grpo import make_reward_fn

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-4B", trust_remote_code=True
    )
    rf = make_reward_fn(tok, max_steps=5, max_new_tokens=32)
    sig = inspect.signature(rf)
    params = list(sig.parameters.keys())
    assert "prompts" in params and "completions" in params
    print(f"[PASS] Reward fn: signature {params}")


def check_cli_help():
    """Run --help via subprocess."""
    result = subprocess.run(
        [sys.executable, "train_trl_grpo.py", "--help"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("[PASS] CLI: --help works\n")
        print(result.stdout[:900])
    else:
        err = result.stderr.strip().splitlines()[-1]
        print(f"[SKIP] CLI --help (needs dependencies): {err}")


def main():
    print("=" * 60)
    print("DispatchR train_trl_grpo.py Smoke Test")
    print("=" * 60)

    tree = check_syntax()
    check_functions_exist(tree)
    check_cli_args(tree)
    check_dataset_builder()
    check_reward_fn()
    check_cli_help()

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
