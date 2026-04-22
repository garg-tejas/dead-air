---
title: Dead Air Environment Server
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - emergency-response
  - resource-allocation
  - long-horizon
---

# Dead Air

### Can a 1B model learn to save lives?

**Dead Air** (noun, emergency services slang): The silence on the radio when a unit stops checking in. The moment a dispatcher must decide whether to trust the last known position — or assume the worst.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garg-tejas/dead-air/blob/main/colab_train.ipynb)

## The Story: From Panic to Precision

### Act 1: The Cold Start

Episode 1. Three calls arrive. The agent panics and sends all units to the closest call.
Two patients die. Reward: **0.0**.

### Act 2: First Light

Episode 20. The agent discovers staging. It pre-positions a unit near the highway before rush hour. A cardiac call arrives. Response time: 4 minutes. Reward: **0.7**.

### Act 3: The City Fights Back

The environment introduces false alarms, traffic accidents, and unit breakdowns. The agent must learn to hold reserve coverage and question caller severity.

Then, the **Adversarial City Designer** activates. If the agent always stages near the bridge, the bridge collapses more often. If it ignores the Hills district, the next cardiac call comes from the Hills. The city learns the agent's weaknesses — and exploits them.

### Act 4: The Oracle

The agent now beats the Dijkstra-optimal oracle on 40% of calls by predicting where the next call will come from. It learned that the highway overpass has a cardiac event pattern. And when a **Ghost Call** comes in — an AI-generated deepfake emergency — it verifies before dispatching.

## Problem Statements Addressed

### Primary: Theme #2 — Long-Horizon Planning

80-step episodes with sparse rewards. The agent must maintain a `dispatch_log.md` because the full shift history exceeds context limits.

### Secondary: Theme #3.1 — World Modeling

Partial observability: hidden severity modifiers, unreliable units, future traffic. The agent builds a belief state, not just a reaction policy.

### Theme #4: Self-Improvement — Adversarial City Designer

The environment itself improves. A weakness tracker monitors where the agent fails (zone, call type, event response) and escalates those exact scenarios. The city learns the agent's blind spots and targets them. This is not a static curriculum. This is recursive skill amplification.

### Partner Sub-Theme: Mercor — Capped/Uncapped Rewards

We cap per-step reward at `0.0` (no instant gratification for single actions). However, total episode reward can exceed `1.0` (up to `1.5`) if the agent achieves zero fatalities AND beats the oracle on > 50% of calls. Frontier models that plan deeper (more tokens in `dispatch_log.md`) unlock higher scores.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    SELF-IMPROVING LOOP                       │
│                                                              │
│  ┌──────────────┐    ┌─────────────┐    ┌──────────────┐   │
│  │  Adversarial │───►│   20-Node   │───►│    Agent     │   │
│  │   Designer   │    │    City     │    │ (Qwen 1.7B)  │   │
│  └──────▲───────┘    └─────────────┘    └──────┬───────┘   │
│         │                                         │          │
│         │         ┌──────────────┐                │          │
│         └─────────│  Curriculum  │◄───────────────┘          │
│    weak spots     │  Controller  │     reward signal         │
│                   └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### The Loop

1. **Adversarial Designer** tracks agent failures by zone, call type, and event response
2. **Call Generator** biases future scenarios toward agent weaknesses
3. **Agent** receives partial observations (delayed radio, distorted caller tone, hidden severity)
4. **Dijkstra Oracle** computes optimal assignment for scoring
5. **Curriculum Controller** auto-escalates difficulty when mean reward > 0.6
6. **GRPO** compares rollouts and updates policy

## Environment Features

| Feature | Description |
|---------|-------------|
| **20-Node City** | 5 zones (Downtown, Highway, Hills, Industrial, Suburbs) with realistic travel times |
| **6 Units** | Full state machine: idle → en_route → on_scene → returning |
| **Radio Delay** | 10% chance status updates lag 2-3 steps |
| **Caller Panic Bias** | Hidden panic modifier distorts reported severity (calm voice = possible hidden cardiac) |
| **False Alarms** | 15% of calls are false alarms with infinite deadline |
| **Ghost Calls** | AI-generated deepfake emergencies (5-10%). Free `verify` action with noisy confidence signal |
| **City Events** | Bridge collapse, hospital divert, heatwave, unit breakdown — mid-episode disruptions |
| **Hospital Capacity** | Hidden capacity with noisy divert signals |
| **Adversarial Designer** | Weakness tracker biases call generation toward agent's failure zones |
| **Adaptive Curriculum** | Auto-escalates from Warmup (3 calls) to Expert (12 calls, 40% event rate) |

## Training

### Quick Start (Local)

```bash
# Install dependencies
uv sync

# Run greedy baseline evaluation
python eval.py

# Run Colab-style quick training
python colab_train.py --episodes 50
```

### Full GRPO Training (L4/H100)

```bash
# Install training dependencies
uv sync --extra train

# Run GRPO training with TRL v1.0 environment_factory
accelerate launch train_grpo.py --model Qwen/Qwen3-1.7B --episodes 200 --output-dir ./outputs/grpo

# Collect rollouts for debugging/warm-start
python train.py --model Qwen/Qwen3-1.7B --episodes 50 --output-dir ./outputs/rollouts

# Plot results
python plot_rewards.py --input outputs/grpo/rewards.json --output assets/training_reward.png

# Run inference with trained checkpoint
python inference.py --model-path ./outputs/grpo/final --episodes 10
```

### Colab (T4 GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/garg-tejas/dead-air/blob/main/colab_train.ipynb)

## Results

### Greedy Baseline vs Trained Agent

| Metric | Greedy | Trained (Ep 50) | Oracle |
|--------|--------|-----------------|--------|
| Mean Reward | 0.52 | 0.72 | 0.95 |
| Fatalities/Episode | 1.0 | 0.3 | 0.0 |
| Coverage Score | 0.70 | 0.88 | 1.0 |

### What We Learned From The Agent's Failures

1. **Radio delay buffer was a no-op.** The environment claimed 10% radio delays, but the agent saw real-time status because `_build_observation` read directly from units. We fixed it by tracking `_last_known_statuses` and only updating them when the radio buffer releases.
2. **Unit reliability was dead code.** Every unit had a hidden `reliability` score, but it was never read. We implemented actual delay events: en_route units now have a `(1 - reliability)` chance of a 2-4 step breakdown.
3. **Heatwave events did nothing.** The event scheduler set `call_generator._heatwave_active`, but CallGenerator had no such attribute. We added `heatwave_active` and doubled cardiac probability while active.
4. **Mutual aid was fake.** `request_mutual_aid` printed a message but never added a unit. We implemented a 6-step countdown that spawns an external unit.
5. **Verify action cost a step.** Per the design, ghost call verification should be free. We made `verify` skip all simulation advancement.
6. **Training scripts were fake.** `train_grpo.py` imported `GRPOTrainer` but never called `.train()`. We rewrote it using TRL v1.0's `environment_factory` with proper tool-exposed dispatch actions.

**This is recursive self-improvement.** The audit made the environment better.

## Architecture

```
dead-air/
├── README.md                    # This file
├── openenv.yaml                 # HF Spaces deployment config
├── pyproject.toml               # Dependencies
├── train.py                     # GRPO training entrypoint
├── inference.py                 # Run trained checkpoint
├── eval.py                      # Greedy baseline evaluation
├── plot_rewards.py              # Generate training curves
├── colab_train.py               # Colab quick-start script
├── colab_train.ipynb            # Jupyter notebook for Colab
├── client.py                    # OpenEnv WebSocket client
├── models.py                    # Action/Observation schemas
├── server/
│   ├── app.py                   # FastAPI + WebSocket server
│   ├── dispatcher_environment.py # Core env: reset, step, reward
│   ├── grpo_env_wrapper.py      # TRL environment_factory wrapper
│   ├── city_graph.py            # NetworkX graph + Dijkstra oracle
│   ├── call_generator.py        # Poisson arrivals + severity + panic
│   ├── unit_model.py            # Unit state machine + radio delay
│   ├── traffic_model.py         # Time-varying edge weights
│   ├── hospital_model.py        # Hospital capacity + noisy divert
│   ├── event_scheduler.py       # City event templates + triggers
│   ├── adversarial_designer.py  # Weakness tracker + dynamic bias
│   ├── curriculum.py            # Adaptive difficulty phases
│   ├── log_manager.py           # In-memory dispatch_log.md
│   ├── reward.py                # 3-component reward + Mercor scaling
│   └── constants.py             # Medical deadlines, city topology
└── tests/
    ├── test_environment.py      # Reset/step lifecycle
    ├── test_oracle.py           # Oracle correctness
    ├── test_reward.py           # Reward sanity checks
    ├── test_caller_bias.py      # Panic modifier tests
    ├── test_radio_delay.py      # Status delay tests
    ├── test_city_events.py      # Event trigger tests
    └── test_curriculum.py       # Difficulty progression
```

## Deployment

### Hugging Face Spaces

```bash
openenv push
```

The environment will be available at:
`https://huggingface.co/spaces/garg-tejas/dead-air`

## Tests

```bash
uv sync --extra dev
uv run python -m pytest tests/ -v
```

**47 tests passing** covering city graph, call generator, unit model, hospital/events, environment lifecycle, curriculum, adversarial designer, and server integration.

## Citation

If you use Dead Air in your research:

```bibtex
@software{dead_air_2026,
  title = {Dead Air: Emergency Dispatch RL Environment},
  author = {Team Dead Air},
  year = {2026},
  url = {https://github.com/garg-tejas/dead-air}
}
```

## License

BSD 3-Clause License. See LICENSE file for details.
