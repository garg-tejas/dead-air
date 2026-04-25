#!/usr/bin/env python3
"""Monitor DispatchR training progress on HF Spaces.

Usage:
    python scripts/monitor_training.py

Shows:
- Current batch, reward, loss
- Curriculum phase
- Recent trajectory samples
- GPU utilization (if nvidia-smi available)
"""

import glob
import json
import os
import subprocess
import sys
import time


def get_latest_log():
    logs = glob.glob("./logs/training_*.log")
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def parse_log_tail(log_file, n=50):
    """Parse the last N lines of the log for key metrics."""
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []

    return lines[-n:]


def extract_metrics(lines):
    """Extract metrics from log lines."""
    metrics = {}
    for line in reversed(lines):
        line = line.strip()
        if "Batch" in line and "/" in line:
            metrics["batch"] = line.split("Batch")[-1].split("---")[0].strip()
        if "Mean reward:" in line:
            try:
                metrics["mean_reward"] = float(line.split("Mean reward:")[1].split("|")[0].strip())
            except (ValueError, IndexError):
                pass
        if "Loss:" in line:
            try:
                metrics["loss"] = float(line.split("Loss:")[1].strip())
            except (ValueError, IndexError):
                pass
        if "Curriculum: phase=" in line:
            try:
                metrics["phase"] = line.split("phase=")[1].split("|")[0].strip()
            except IndexError:
                pass
        if "CURRICULUM ESCALATED" in line:
            metrics["escalation"] = line
        if "Saved checkpoint" in line:
            metrics["last_checkpoint"] = line.split("Saved checkpoint to")[-1].strip()
        if "Pushed checkpoint" in line:
            metrics["last_push"] = line.split("Pushed checkpoint to")[-1].strip()
    return metrics


def get_gpu_info():
    """Get GPU utilization via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_trajectory_stats():
    """Get stats from trajectory file if it exists."""
    traj_file = "./outputs/unsloth_grpo/trajectory.jsonl"
    if not os.path.exists(traj_file):
        return None

    try:
        # Count lines and get last episode
        with open(traj_file, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        total_steps = len(lines)
        last_record = json.loads(lines[-1])

        return {
            "total_steps": total_steps,
            "last_batch": last_record.get("batch"),
            "last_episode": last_record.get("episode"),
            "last_reward": last_record.get("reward"),
        }
    except (json.JSONDecodeError, KeyError):
        return None


def main():
    print("=" * 60)
    print("DispatchR Training Monitor")
    print("=" * 60)
    print("")

    # Check if training is running
    pid_file = "./training.pid"
    if os.path.exists(pid_file):
        with open(pid_file) as f:
            pid = f.read().strip()
        # Check if process exists
        try:
            os.kill(int(pid), 0)
            print(f"Training status: RUNNING (PID {pid})")
        except (OSError, ValueError):
            print("Training status: NOT RUNNING (stale PID file)")
    else:
        print("Training status: NOT STARTED")
    print("")

    # Get latest log
    log_file = get_latest_log()
    if not log_file:
        print("No training logs found yet.")
        return

    print(f"Log file: {log_file}")
    print("")

    # Parse metrics
    lines = parse_log_tail(log_file, n=100)
    metrics = extract_metrics(lines)

    if metrics:
        print("Latest Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("No metrics found in log yet.")
    print("")

    # Trajectory stats
    traj_stats = get_trajectory_stats()
    if traj_stats:
        print("Trajectory Stats:")
        for key, value in traj_stats.items():
            print(f"  {key}: {value}")
        print("")

    # GPU info
    gpu_info = get_gpu_info()
    if gpu_info:
        print("GPU Status:")
        for line in gpu_info.split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                print(f"  GPU {parts[0]}: {parts[1]}")
                print(f"    Temp: {parts[2]}°C | Util: {parts[3]}%")
                print(f"    Memory: {parts[4]}/{parts[5]} MiB")
        print("")
    else:
        print("GPU Status: nvidia-smi not available")
        print("")

    # Show recent log lines
    print("Recent Log Lines:")
    for line in lines[-10:]:
        print(f"  {line.rstrip()}")
    print("")

    print("=" * 60)
    print("Refresh with: python scripts/monitor_training.py")
    print("Tail live with: tail -f logs/training_*.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
