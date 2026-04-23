# Dead Air — Project Status & Handoff Document

> **Created**: 2026-04-23  
> **Purpose**: Comprehensive summary of build progress, current blockers, and recommended fixes for anyone continuing this project.  
> **Target Platform**: Lightning AI Studio with L4 GPU (24GB VRAM)  
> **Remote Repo**: https://github.com/garg-tejas/dead-air

---

## 1. What We Built

**Dead Air** is a complete OpenEnv-compliant reinforcement learning environment simulating emergency medical dispatch. It was built for the Meta OpenEnv Hackathon (India, Apr 2026).

### Core Features
- **20-node canonical city graph** with Dijkstra shortest-path oracle
- **6-unit fleet** with state machine (idle, en_route, on_scene, returning, delayed)
- **Radio delay** — unit statuses are stale by 1-3 steps (POMDP)
- **Call generator** — Poisson arrivals with panic bias, false alarms, ghost calls (deepfakes)
- **Hospital capacity** — hidden true capacity, noisy reported status
- **City events** — bridge collapse, heatwave, unit breakdown, hospital divert
- **Adversarial designer** — tracks model weaknesses and biases future calls
- **Adaptive curriculum** — auto-escalates/de-escalates difficulty based on rolling mean reward
- **3-component reward** — survival probability + response time + coverage, with Mercor scaling
- **Verify action** — free background check on calls with noisy confidence (high/medium/low)
- **Full OpenEnv server** — FastAPI + WebSocket, deployable to Hugging Face Spaces

### File Inventory

```
dead-air/
├── README.md                          # Full story arc, architecture diagram, usage
├── HF_BLOG.md                         # Mini-blog outline for HuggingFace submission
├── PLAN.md                            # Original hackathon plan
├── openenv.yaml                       # HF Spaces deployment config
├── pyproject.toml                     # Dependencies
│
├── models.py                          # Pydantic schemas (DispatchAction, etc.)
├── client.py                          # EmergencyDispatcherClient (OpenEnv WebSocket)
├── eval.py                            # Greedy baseline evaluation
├── diagnose.py                        # Per-episode diagnostic script
├── demo.py                            # Interactive terminal demo
├── colab_train.py                     # Colab quick-start script
├── colab_train.ipynb                  # Jupyter notebook for Google Colab
├── train.py                           # Rollout collection (legacy)
├── train_grpo.py                      # TRL GRPO trainer with environment_factory
├── train_unsloth_grpo.py              # NEW — Unsloth-based GRPO (not yet tested end-to-end)
├── plot_rewards.py                    # Reward curve visualization
├── inference.py                       # Run trained checkpoint
├── generate_assets.py                 # Architecture diagram generator
│
├── server/
│   ├── app.py                         # FastAPI + WebSocket OpenEnv server
│   ├── __init__.py
│   ├── grpo_env_wrapper.py            # DeadAirGRPOEnv for TRL environment_factory
│   ├── unsloth_grpo_utils.py          # GRPO loss, generation, reward shaping for Unsloth
│   ├── rollout_utils.py               # Shared format_prompt, parse_action, collect_rollout
│   ├── constants.py                   # 20-node city, deadlines, unit configs, curriculum phases
│   ├── city_graph.py                  # NetworkX graph + Dijkstra oracle
│   ├── traffic_model.py               # Time-varying edge weights + accident injection
│   ├── unit_model.py                  # Unit state machine + RadioDelayBuffer
│   ├── call_generator.py              # Poisson arrivals + panic + false alarms + ghosts
│   ├── hospital_model.py              # Hidden capacity + noisy divert signals
│   ├── event_scheduler.py             # Bridge collapse, heatwave, breakdown, divert
│   ├── log_manager.py                 # In-memory dispatch_log.md
│   ├── dispatcher_environment.py      # Core env: reset/step/reward
│   ├── adversarial_designer.py        # Weakness tracker + dynamic call bias
│   ├── curriculum.py                  # Auto-escalate/de-escalate difficulty
│   └── reward.py                      # 3-component reward + Mercor scaling
│
├── tests/                             # 47 passing unit tests
│   ├── test_city_graph.py             # 8 tests
│   ├── test_call_generator.py         # 7 tests
│   ├── test_unit_model.py             # 10 tests
│   ├── test_hospital_events.py        # 8 tests
│   ├── test_environment.py            # 9 tests
│   └── test_connection.py             # 6 tests
│
└── outputs/                           # Training checkpoints (gitignored)
```

---

## 2. Architecture

### Environment Loop
```
GRPOTrainer
    └── environment_factory=DeadAirGRPOEnv
            ├── reset(difficulty) → initial prompt
            └── step(completion_text) → parsed action → env.step() → next observation
```

### Action Format
The model outputs free-form text. Our parser looks at the **last line** of the completion:
```
Think step by step about which calls are most urgent...

dispatch(unit_id=1, call_id=3)
```

Supported actions: `dispatch`, `reroute`, `stage`, `divert`, `verify`, `request_mutual_aid`, `log`, `hold`.

### Reward Function
```python
reward = survival_probability * (1 - response_time_penalty) * coverage_factor
```
- **Survival**: Based on call type + elapsed time (cardiac has steep decay)
- **Response time**: Penalty for time to first dispatch
- **Coverage**: Ratio of handled calls to total calls

### Arrival Bonus (New)
- **+0.03 per unit** that successfully arrives on scene
- **Disabled for first 20 episodes**
- **Auto-enabled only if** rolling reward variance < 0.01 (flat curve detection)
- Capped so total reward ≤ 1.0

---

## 3. Training History

### Attempt 1: TRL `train_grpo.py` (Original)
- **Model**: Qwen/Qwen3-1.7B
- **Issue**: vLLM CUDA conflicts (`libcusparseLt.so.0` error)
- **Fix**: Added `--no-vllm` flag
- **Issue 2**: Model generated free-form text, never called tools → reward = 0
- **Fix**: Added `step(text)` parser to `DeadAirGRPOEnv` + explicit action syntax in prompt

### Attempt 2: TRL `train_grpo.py` (Fixed Parser)
- **Model**: Qwen/Qwen3-1.7B → Qwen/Qwen3.5-2B
- **Issue**: `completions/clipped_ratio: 0.9375` at 512 tokens
- **Fix**: Increased `max_completion_length` to 1536
- **Issue 2**: Chat template required `user` message, not just `system`
- **Fix**: Added user message to conversational dataset

### Attempt 3: TRL `train_grpo.py` (Working but Slow)
- **Status**: ✅ Runs successfully on Lightning AI L4
- **Speed**: ~5 min per step, ~2 hours per 50-episode batch
- **Problem**: Too slow for meaningful training (200 episodes = ~8 hours)

### Attempt 4: Unsloth `train_unsloth_grpo.py` (CURRENT BLOCKER)
- **Goal**: 2-5× speedup via Unsloth 4-bit kernels
- **Status**: ❌ Cannot run — dependency hell

---

## 4. Current Blocker: Dependency Hell

### Root Cause
Installing `unsloth` into the Lightning AI `cloudspace` conda environment pulled in ~100 packages with conflicting version constraints, breaking the previously working TRL setup.

### Chain of Failures

| Step | Action | Result |
|------|--------|--------|
| 1 | `pip install unsloth` | Installed unsloth, but also upgraded numpy to 2.4.4 |
| 2 | `pip install -e .` (to get `openenv-core`) | Upgraded torch 2.10→2.11, torchvision 0.25→0.26, datasets 4.3→4.8 |
| 3 | `pip install numpy==1.26.4` | Fixed numpy, but torch 2.11 now incompatible with cuda-bindings |
| 4 | `pip install torch==2.10.0` | Downgraded torch, but torchvision 0.26 still incompatible |
| 5 | `pip install torchvision==0.25.0` | Fixed torchvision, but vllm 0.19 now broken (compiled against torch 2.11) |
| 6 | `pip uninstall vllm` | TRL still imports vllm modules unconditionally → crash |
| 7 | `pip install mergekit` | TRL needs mergekit for callbacks |
| 8 | `pip install llm-blender` | llm-blender incompatible with transformers 5.5 (`TRANSFORMERS_CACHE` removed) |

### Current State of `cloudspace` Env
- **torch**: 2.10.0 (working)
- **torchvision**: 0.25.0 (working)
- **numpy**: 1.26.4 (working)
- **transformers**: 5.5.0 (working)
- **trl**: 0.24.0 (imports broken due to missing vllm/mergekit/llm_blender)
- **vllm**: uninstalled (was 0.19.2rc1, compiled against torch 2.11)
- **unsloth**: 2026.4.7 (imports work, but training script untested)

### What Breaks Now

**`train_grpo.py`**: ❌
```
RuntimeError: Failed to import trl.trainer.grpo_trainer
→ No module named 'llm_blender'
→ llm_blender tries to import TRANSFORMERS_CACHE from transformers.utils.hub (removed in transformers 5.5)
```

**`train_unsloth_grpo.py`**: ❌
```
ImportError: No module named 'unsloth'
→ Wait, unsloth IS installed. But the script runs via `uv run` which uses a DIFFERENT venv.
→ Need to use `/system/conda/miniconda3/envs/cloudspace/bin/python` directly.
```

---

## 5. Working Configurations

### Configuration A: Fresh Lightning AI Instance (RECOMMENDED)
If you can create a new Lightning AI Studio instance:

```bash
# 1. Clone repo
git clone https://github.com/garg-tejas/dead-air.git
cd dead-air

# 2. The cloudspace env already has unsloth installed.
#    Do NOT run `pip install -e .` — it will upgrade torch and break everything.
#    Instead, manually install ONLY the missing packages:

/system/conda/miniconda3/envs/cloudspace/bin/pip install openenv-core==0.2.3
/system/conda/miniconda3/envs/cloudspace/bin/pip install networkx==3.6.1
/system/conda/miniconda3/envs/cloudspace/bin/pip install matplotlib==3.8.2
/system/conda/miniconda3/envs/cloudspace/bin/pip install pydantic==2.10.6
/system/conda/miniconda3/envs/cloudspace/bin/pip install jmespath==1.1.0
/system/conda/miniconda3/envs/cloudspace/bin/pip install peft==0.19.1

# 3. Run Unsloth training
/system/conda/miniconda3/envs/cloudspace/bin/python train_unsloth_grpo.py \
    --model unsloth/Qwen3.5-2B \
    --episodes 200 \
    --batch-size 8 \
    --output-dir ./outputs/unsloth_grpo \
    --save-every 50
```

### Configuration B: Fix Current Instance
If you're stuck on the current instance:

```bash
# Option 1: Reinstall TRL with compatible versions
/system/conda/miniconda3/envs/cloudspace/bin/pip uninstall -y trl
/system/conda/miniconda3/envs/cloudspace/bin/pip install trl==0.24.0 --no-deps
# Then manually install trl's deps one by one, avoiding anything that upgrades torch

# Option 2: Downgrade transformers to 4.56 (might fix llm_blender)
/system/conda/miniconda3/envs/cloudspace/bin/pip install transformers==4.56.0
# WARNING: This may break other things.

# Option 3: Patch llm_blender locally
# Edit /system/conda/miniconda3/envs/cloudspace/lib/python3.12/site-packages/llm_blender/blender/blender_utils.py
# Change:
#   from transformers.utils.hub import TRANSFORMERS_CACHE
# To:
#   from transformers.utils.hub import default_cache_path as TRANSFORMERS_CACHE
```

### Configuration C: Colab (Guaranteed Working)
The `colab_train.ipynb` notebook was tested and works on Google Colab T4 GPU.

---

## 6. Unsloth Training Script (`train_unsloth_grpo.py`)

### What It Does
- Loads `unsloth/Qwen3.5-2B` in **4-bit quantization** (~1.5 GB)
- Adds LoRA adapters (r=16, alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
- Runs **custom multi-turn GRPO loop**:
  1. Batch of N environments reset
  2. For each of 25 steps: generate completions → parse actions → step envs
  3. Compute episode rewards (with optional arrival bonus)
  4. GRPO update: normalize advantages, recompute log-probs, clipped surrogate loss
- Saves checkpoints every `--save-every` episodes

### Key Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `unsloth/Qwen3.5-2B` | HF model ID |
| `--episodes` | 200 | Total training episodes |
| `--batch-size` | 8 | Parallel envs (safe for L4 24GB) |
| `--max-new-tokens` | 512 | Max tokens per action |
| `--learning-rate` | 5e-6 | AdamW LR |
| `--save-every` | 50 | Checkpoint frequency |

### Expected Speed
| Setup | Time per 50 episodes | Time per 200 episodes |
|-------|---------------------|----------------------|
| TRL (`train_grpo.py`) | ~2 hours | ~8 hours |
| Unsloth (`train_unsloth_grpo.py`) | ~20-40 min | ~1.5-3 hours |

### Known Issues with `train_unsloth_grpo.py`
1. **Untested end-to-end** — the script was written but never ran successfully due to env issues
2. **GRPO loss computation** may have bugs — validate against TRL's GRPO on a simple task first
3. **Reward signal** — all previous TRL runs showed `reward: 0` because episodes didn't complete in small test batches. With 200 episodes and batch_size=8, most episodes should hit MAX_STEPS=25 and produce non-zero rewards.

---

## 7. Testing

### Run Unit Tests
```bash
cd /home/zeus/content  # or wherever repo is cloned
python -m pytest tests/ -q
```
**Expected**: 47 passed

### Run Greedy Baseline
```bash
python eval.py --episodes 100 --difficulty learning
```
**Expected**: Mean reward ~0.47-0.53

### Run Demo
```bash
python demo.py --episodes 1 --difficulty learning
```

---

## 8. Submission Checklist (Hackathon)

Per the judging criteria:

- [x] **OpenEnv compliant** — `server/app.py` implements reset/step/state
- [x] **HF Space config** — `openenv.yaml` present
- [ ] **Training script** — `train_unsloth_grpo.py` ready, but NEEDS SUCCESSFUL RUN
- [ ] **Reward curves** — Need to generate and commit `reward_curve.png`
- [ ] **HF Space deployed** — Need to `openenv push`
- [ ] **Mini-blog/video** — `HF_BLOG.md` is an outline, needs publishing
- [ ] **README with links** — Need Space URL + blog link

---

## 9. Quick Commands Reference

```bash
# SSH into Lightning AI
ssh s_01kpv0em7j72yxfatcdmzwhqaq@ssh.lightning.ai

# Repo location
ls /home/zeus/content

# Use cloudspace Python directly
PYTHON=/system/conda/miniconda3/envs/cloudspace/bin/python
$PYTHON --version  # Python 3.12.3

# Check CUDA
$PYTHON -c "import torch; print(torch.cuda.is_available())"

# Check Unsloth
$PYTHON -c "import unsloth; print(unsloth.__version__)"

# Run training (once env is fixed)
$PYTHON train_unsloth_grpo.py \
    --model unsloth/Qwen3.5-2B \
    --episodes 200 \
    --batch-size 8 \
    --output-dir ./outputs/unsloth_grpo \
    --save-every 50

# View GPU usage
nvidia-smi

# Check what's installed
$PYTHON -m pip list | grep -E "torch|transformers|trl|unsloth|peft|datasets"
```

---

## 10. Recommended Next Steps

1. **Fix the environment** (see Section 5)
2. **Run a small test** (10 episodes) to verify `train_unsloth_grpo.py` works
3. **Scale to 200 episodes** for real training
4. **Generate plots** with `plot_rewards.py`
5. **Deploy to HF Space** with `openenv push`
6. **Publish mini-blog** on HuggingFace

---

## 11. Contact / Context

- **Hackathon**: Meta OpenEnv Hackathon, India, Apr 25-26 2026
- **Theme**: #3.1 Professional Tasks (World Modeling) + #2 Long-Horizon Planning
- **Team**: garg-tejas (solo)
- **Model**: Qwen3.5-2B (2B params, instruct, non-thinking by default)
- **GPU**: Lightning AI L4 (24GB VRAM)
- **Key libraries**: Unsloth, TRL, Transformers, OpenEnv, NetworkX

---

*Last updated: 2026-04-23 by OpenCode (AI assistant)*
