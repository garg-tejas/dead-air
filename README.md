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

**Dead Air** (noun, emergency services slang): The silence on the radio when a unit stops checking in. The moment a dispatcher must decide whether to trust the last known position — or assume the worst.

A reinforcement learning environment for emergency medical dispatch, built for the Meta OpenEnv Hackathon (India, April 2026).

**Status**: 47/47 tests passing. Active training pipeline validated on Lightning AI L4.

## What It Is

Dead Air simulates an 8-hour shift for an emergency dispatch commander in a 20-node city. The agent must:

- Dispatch 6 units to cardiac, trauma, and fire emergencies
- Deal with radio delays (statuses lag 2-3 steps behind reality)
- Verify ghost calls (AI-generated false emergencies)
- Respond to city events (bridge collapse, heatwave, hospital divert, unit breakdown)
- Maintain coverage across 5 zones so no area is left uncovered

The environment is a **partially-observable Markov decision process (POMDP)** with hidden severity modifiers, noisy caller tones, and dynamic traffic conditions.

## Problem Statements

### Primary: Theme #2 — Long-Horizon Planning

80-step episodes with sparse end-of-episode rewards. The agent must plan across the full shift, not just react to the current call.

### Secondary: Theme #3.1 — World Modeling

Partial observability: the agent sees delayed unit statuses, caller tone (not true severity), and reported hospital capacity (not true capacity). It must build a belief state from incomplete information.

### Theme #4: Self-Improvement — Adversarial City Designer

A weakness tracker monitors where the agent fails (by zone, call type, and event response) and biases future scenarios toward those exact failure modes. The environment itself adapts to exploit the agent's blind spots.

### Partner Sub-Theme: Mercor — Capped/Uncapped Rewards

Episode reward is capped at 1.0 for most cases, but can reach 1.5 with the Mercor bonus if the agent achieves zero fatalities AND mean response time exceeds 50% of oracle performance.

## Environment Features

| Feature | Description |
|---------|-------------|
| **20-Node City** | 5 zones (Downtown, Highway, Hills, Industrial, Suburbs) with Dijkstra shortest-path routing |
| **6 Units** | State machine: idle → en_route → on_scene → returning |
| **Radio Delay** | 10% chance status updates lag 2-3 steps (POMDP) |
| **Caller Panic Bias** | Hidden panic modifier distorts reported severity; calm voice can mask cardiac |
| **False Alarms** | 10-20% of calls are false alarms (infinite deadline, no penalty for ignoring) |
| **Ghost Calls** | AI-generated deepfake emergencies (0-10%). `verify` action returns noisy confidence |
| **City Events** | Bridge collapse (reroutes traffic), hospital divert, heatwave (2x cardiac rate), unit breakdown |
| **Hospital Capacity** | Hidden true capacity; reported status is noisy |
| **Adversarial Designer** | Weakness tracker biases call generation toward agent failure zones |
| **Adaptive Curriculum** | Auto-escalates from Warmup (3 calls, 0 events) to Expert (12 calls, 40% event rate). Now controllable via CLI in training |
| **LLM-Friendly Prompts** | Active Calls section explicitly shows `(none)` when empty; system prompt includes inline examples and conciseness rules to reduce rambling |

## Architecture

```
dead-air/
├── README.md                    # This file
├── openenv.yaml                 # HF Spaces deployment config
├── pyproject.toml               # Package dependencies
│
├── train_unsloth_grpo.py        # PRIMARY: Unsloth GRPO training (2-5x faster)
├── train_grpo.py                # FALLBACK: Manual GRPO (standard transformers)
├── eval.py                      # Greedy baseline evaluation
├── demo.py                      # Interactive terminal demo
├── diagnose.py                  # Per-episode diagnostic script
├── inference.py                 # Run trained checkpoint
├── plot_rewards.py              # Reward curve visualization
├── client.py                    # OpenEnv WebSocket client
├── models.py                    # Pydantic Action/Observation schemas
│
├── scripts/
│   ├── launch_hf_training.py    # HF Spaces training launcher (with nohup)
│   ├── launch_hf_training.sh    # Bash version of launcher
│   └── monitor_training.py      # Remote training monitor
│
├── server/
│   ├── app.py                   # FastAPI + WebSocket OpenEnv server
│   ├── dispatcher_environment.py # Core env: reset, step, reward
│   ├── grpo_env_wrapper.py      # Wrapper for LLM training (JSON parser + metrics)
│   ├── unsloth_grpo_utils.py    # GRPO loss computation for batched token trajectories
│   ├── city_graph.py            # NetworkX graph + Dijkstra oracle
│   ├── call_generator.py        # Poisson arrivals + panic + false alarms + ghosts
│   ├── unit_model.py            # Unit state machine + RadioDelayBuffer
│   ├── traffic_model.py         # Time-varying edge weights + accident injection
│   ├── hospital_model.py        # Hidden capacity + noisy divert signals
│   ├── event_scheduler.py       # City event templates + random triggers
│   ├── adversarial_designer.py  # Weakness tracker + dynamic bias
│   ├── curriculum.py            # Auto-escalate/de-escalate difficulty
│   ├── reward.py                # 3-component reward + Mercor scaling
│   └── constants.py             # Medical deadlines, city topology, unit configs
│
└── tests/
    ├── test_environment.py      # Env lifecycle (reset, step, dispatch, hold)
    ├── test_call_generator.py   # Call generation, panic, false alarms, ghosts
    ├── test_unit_model.py       # Unit state machine + radio delay buffer
    ├── test_city_graph.py       # Graph topology + oracle assignment
    ├── test_hospital_events.py  # Hospital model + event scheduler
    └── test_connection.py       # End-to-end episode + curriculum
```

## Quick Start

### Deploy on Hugging Face Spaces (Recommended for Hackathon)

See the complete setup guide: [`HF_SETUP_GUIDE.md`](HF_SETUP_GUIDE.md)

```bash
# One-command launch on HF Spaces A100
python scripts/launch_hf_training.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --hub-model-id yourname/dead-air-grpo
```

### 1. Run Tests

```bash
python -m pytest tests/
# 47 passed in ~0.5s
```

### 2. Greedy Baseline

```bash
python eval.py
```

Runs 10 episodes with a greedy dispatcher (closest idle unit to highest-priority call). Expected mean reward: **~0.45–0.55**.

### 3. Interactive Demo

```bash
python demo.py --episodes 3 --difficulty learning --delay 0.3
```

Watch the greedy agent handle a full shift in real time.

### 4. Diagnostics

```bash
python diagnose.py --episodes 10 --agent greedy
```

Per-episode breakdown: reward, fatalities, action histogram, events.

## Training

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on L4 24GB)
- `unsloth`, `transformers`, `torch`, `trl`

```bash
pip install unsloth transformers torch trl numpy networkx
```

> **Note**: `import unsloth` must happen **before** `import transformers` to enable kernel optimizations.

### Primary: Unsloth GRPO Training

```bash
python train_unsloth_grpo.py \
  --model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --max-completion-length 1536 \
  --curriculum \
  --save-every 25 \
  --trajectory-file ./outputs/unsloth_grpo/trajectory.jsonl \
  --output-dir ./outputs/unsloth_grpo
```

**Why Unsloth?**
- 2–5× faster than manual GRPO on L4
- Pre-quantized `-bnb-4bit` models load in ~30s (no CPU RAM spike)
- Thinking-enabled models generate `<think>` reasoning blocks that RL can optimize

**Key flags:**
- `--model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit`: Pre-quantized 4B thinking model (fits in ~17GB)
- `--model unsloth/Qwen3-14B-unsloth-bnb-4bit`: Upgrade to 14B on A100 (recommended)
- `--max-completion-length 1536`: Room for reasoning traces + JSON action
- `--curriculum`: Enable performance-gated difficulty escalation
- `--curriculum-phases`: Custom phase sequence (default: `warmup,learning,advanced,expert`)
- `--curriculum-min-episodes 30`: Min episodes before escalation
- `--curriculum-escalate-threshold 0.65`: Mean reward threshold to advance
- `--epsilon-start 1.0`: First batches are 100% greedy actions (provides learning signal)
- `--trajectory-file`: Saves every prompt/completion as JSONL for auditing
- `--push-to-hub --hub-model-id yourname/model`: Push checkpoints to HF Hub automatically

**Expected behavior (from 8-episode pilot):**
- Batch 1 (ε=1.0): Mean reward ~0.60–0.80, all actions are greedy dispatch
- With active calls: Model outputs valid JSON 100% of the time
- With no calls: Improved system prompt reduces rambling; expect valid `hold` JSON >80%
- Curriculum escalates when 3-batch mean reward ≥ 0.65 after 30+ episodes in phase

### Training on Hugging Face Spaces (Recommended)

For the hackathon, HF Spaces GPU gives better hardware and persistent checkpoint storage:

```bash
export HF_TOKEN=hf_...

# Launch with 14B model on A100 (best results)
python scripts/launch_hf_training.py \
  --model unsloth/Qwen3-14B-unsloth-bnb-4bit \
  --episodes 200 \
  --batch-size 8 \
  --hub-model-id yourname/dead-air-grpo

# Monitor remotely
python scripts/monitor_training.py
tail -f logs/training_*.log
```

**GPU selection:**
| GPU | VRAM | $/hr | Max Model |
|-----|------|------|-----------|
| L4 | 24GB | $0.80 | 4B–8B |
| L40S | 48GB | $1.80 | 8B–14B |
| **A100** | **80GB** | **$2.50** | **14B–30B** |

**A100 is recommended** for the hackathon: 80GB fits 14B with headroom, 2.4× memory bandwidth of L4, and $60 buys exactly 24 hours.

See [`scripts/README.md`](scripts/README.md) for full HF Spaces deployment guide.

### Fallback: Manual GRPO Training

If Unsloth is unavailable:

```bash
python train_grpo.py \
  --model Qwen/Qwen3.5-2B \
  --episodes 200 \
  --batch-size 8 \
  --use-4bit \
  --trajectory-file ./outputs/trajectory.json \
  --output-dir ./outputs/grpo
```

Same `--curriculum` flags supported.

### Monitoring

Track these during training:
- `Mean reward` — should increase from ~0.55 toward ~0.70+
- `Non-zero` — fraction of episodes with reward > 0
- `Loss` — should be non-zero and gradually decrease
- `Curriculum: phase=X` — shows current difficulty phase
- `🎓 CURRICULUM ESCALATED` — confirms progression

After training, inspect the trajectory file:
```bash
python -c "
import json
with open('outputs/unsloth_grpo/trajectory.jsonl') as f:
    for line in f:
        d = json.loads(line)
        print(d['batch'], d['episode'], d['completion'][:100])
        break
"
```

## Reward Design

The episode reward has 4 components:

1. **Response Score** (50%): Ratio of oracle response time to actual response time. Capped at 1.0.
2. **Fatality Component** (30%): 1.0 if zero deaths, penalized by 0.5 per fatality.
3. **Coverage Score** (20%): Time-averaged fraction of steps where all 5 zones had a unit within 10 minutes.
4. **Action Validity Shaping**: Small bonus for valid actions, penalty for invalid actions and excessive holding.

```
reward = 0.50 * response_score + 0.30 * fatality_component + 0.20 * coverage_score
         + validity_bonus - invalidity_penalty - idle_penalty
```

Plus Mercor bonus (+0.5) if fatalities == 0 and response_score > 0.5.

## How the Environment Prevents Reward Hacking

| Safeguard | How It Works |
|-----------|-------------|
| **Invalid action penalty** | Every invalid dispatch/stage/verify costs -0.02 in final reward |
| **Idle penalty** | Every `hold` step costs -0.005; encourages active dispatching |
| **Time-averaged coverage** | Coverage is measured per-step, not just at episode end. Can't cluster units at the end |
| **Oracle uses start locations** | Oracle assignments are computed from unit start positions, not scattered end-state positions |
| **Bridge collapse updates graph** | `CityGraph` recomputes all shortest paths when edges are destroyed. Units actually reroute |
| **Radio delay is visible** | `last_update_step` shows when status was last confirmed, not current step |
| **Ghost calls are detectable** | `verify` returns high confidence only 40% of the time on ghost calls (not 60% as design spec said) |

## OpenEnv Compliance

This environment implements the OpenEnv interface:

- `reset(difficulty)` → initial observation
- `step(action_dict)` → next observation, reward at episode end
- `state` → `State(episode_id, step_count)`
- `get_ground_truth()` → full episode metrics for analysis

Deploy to Hugging Face Spaces:
```bash
openenv push
```

## Judging Metrics

After each episode, `get_ground_truth()` exposes:

```json
{
  "fatality_count": 2,
  "valid_action_rate": 0.65,
  "invalid_action_rate": 0.10,
  "hold_rate": 0.25,
  "dispatch_rate": 0.40,
  "avg_response_time": 4.5,
  "calls_missed": 2,
  "response_score": 0.72,
  "coverage_score": 0.85
}
```

These metrics let judges verify that improvement is real, not just reward hacking.

## Known Limitations

- **Training time**: ~15–20 min per batch of 8 episodes on L4. 200 episodes ≈ 6–8 hours.
- **vLLM not used**: Batched `transformers.generate()` is used instead because vLLM's KV cache conflicts with training memory on 24GB. Unsloth kernels mitigate the speed penalty.
- **Fire scene time longer**: Fire calls tie up units for 5 steps (vs 3 for cardiac/trauma). This is correct but reduces effective fleet size during fire-heavy episodes.
- **Model occasionally echoes prompt**: Rarely repeats system prompt text instead of generating an action when confused (mitigated by improved prompt with explicit examples).

## Team

- **Solo project**: garg-tejas
- **Model**: unsloth/Qwen3-4B-Thinking-2507-bnb-4bit (4B params, thinking-enabled, pre-quantized)
- **GPU**: Lightning AI L4 (24GB VRAM)
- **Hackathon**: Meta OpenEnv Hackathon, India, April 25-26 2026
- **Tests**: 47/47 passing

## License

BSD 3-Clause License. See LICENSE file for details.
