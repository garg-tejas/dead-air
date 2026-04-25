# DispatchR Project Guide

## 1. Overview

DispatchR is a reinforcement learning (RL) environment and training stack for emergency dispatch optimization. It simulates a city-wide emergency response system where an agent must allocate limited units under uncertainty and changing operational conditions.

Core goals:

- Improve response speed for urgent incidents.
- Reduce preventable fatalities.
- Maintain city-wide coverage over a full operational shift.
- Train robust policies that adapt to disruptions and noisy information.

The project is designed around OpenEnv for environment serving and integrates GRPO-style training pipelines, with Unsloth as the primary high-performance training path.

## 2. Problem Statement

Emergency dispatch is a long-horizon, partially observable decision problem.

In each episode, the policy must:

- Handle incoming calls (cardiac, trauma, fire, false alarms, ghost calls).
- Assign and re-assign a finite set of response units.
- Operate with delayed or noisy status updates (radio delay, caller uncertainty).
- Respond to city events (traffic changes, hospital divert, breakdowns, heatwave effects).

Decisions made early in the episode can cause downstream consequences, so the policy must plan beyond immediate local gains.

## 3. High-Level Architecture

### 3.1 Main Layers

1. Environment API Layer

- `server/app.py`: OpenEnv/FastAPI app construction and serving entrypoint.

2. Core Simulation Layer

- `server/dispatcher_environment.py`: episode reset, step loop, action handling, observation construction, and end-of-episode reward integration.

3. Domain Subsystems

- `server/city_graph.py`: graph topology, travel-time oracle, shortest paths.
- `server/call_generator.py`: stochastic incident generation with hidden modifiers.
- `server/unit_model.py`: unit state machine and radio delay buffering.
- `server/traffic_model.py`: time-varying or event-driven traffic effects.
- `server/hospital_model.py`: reported vs hidden hospital capacity/divert dynamics.
- `server/event_scheduler.py`: disruptive city event scheduling and application.
- `server/curriculum.py`: auto difficulty progression.
- `server/adversarial_designer.py`: biases scenarios toward policy weakness zones.

4. Reward and Evaluation Layer

- `server/reward.py`: response/fatality/coverage reward computation.
- `eval.py`, `diagnose.py`: baseline and analysis scripts.

5. Training Layer

- `train_unsloth_grpo.py`: primary Unsloth-optimized GRPO training.
- `train_grpo.py`: fallback standard Transformers/PEFT training.
- `server/grpo_env_wrapper.py`: text action parsing and tool-style environment wrapper.

6. Validation Layer

- `tests/`: subsystem and integration tests.

### 3.2 Deployment/Utility Layer

- `openenv.yaml`: OpenEnv deployment config.
- root `app.py`: lightweight health-check HTTP app for training container readiness.
- `scripts/`: launch and monitor training workflows.

## 4. Repository Structure (Practical View)

Top-level files:

- `README.md`: project summary, quickstart, training notes.
- `pyproject.toml`: package metadata and dependencies.
- `main.py`: placeholder script.
- `models.py`: action/observation/state schemas.
- `client.py`: environment client utility.

Primary directories:

- `server/`: core environment and domain logic.
- `tests/`: automated tests.
- `docs/`: research and project documentation.
- `scripts/`: training launch and monitoring helpers.
- `assets/`: generated/static assets.

## 5. Environment Design

### 5.1 Episode Framing

- Episode length: `MAX_STEPS = 80` (full shift abstraction).
- Initial resources: default unit set from `server/constants.py`.
- Difficulty phases: warmup -> learning -> advanced -> expert.

### 5.2 Observability Model

The environment is intentionally partially observable:

- Unit statuses can be stale due to radio delay.
- Caller tone is observable, true severity modifiers are hidden.
- Hospital reported status may not match true capacity.
- Ghost/false incidents create signal ambiguity.

### 5.3 Agent Actions

Supported action types include:

- `dispatch`
- `reroute`
- `stage`
- `request_mutual_aid`
- `divert`
- `verify`
- `log`
- `hold`

Invalid actions are tracked and penalized to discourage degenerate behavior.

### 5.4 World Dynamics

At each step, the environment performs:

1. Action processing and validity accounting.
2. Unit updates (movement, on-scene resolution, delays, outages).
3. Radio buffer release of delayed status updates.
4. Call progression and new call generation.
5. Event checks and event effect application.
6. Mutual aid arrival updates.
7. Coverage accumulation and episode-end checks.

## 6. Reward System

Episode reward is a weighted blend of operational outcomes:

- Response score (actual vs oracle travel-time performance)
- Fatality component
- Coverage score

Then adjusted by action-quality shaping:

- Valid-action bonus
- Invalid-action penalty
- Idle/over-hold penalty

This design helps prevent reward hacking and forces meaningful dispatch decisions across the episode horizon.

## 7. Data Model and Interfaces

`models.py` defines core schemas:

- `DispatchAction`
- `DispatchObservation`
- `EpisodeGroundTruth`
- `DispatcherState`

OpenEnv interface alignment:

- `reset(difficulty)`
- `step(action)`
- `state`
- `get_ground_truth()`

The simulator can be served and consumed by compatible clients/tools with consistent contracts.

## 8. Training Pipelines

### 8.1 Primary: Unsloth GRPO

`train_unsloth_grpo.py` is the preferred path.

Benefits:

- Better throughput on constrained GPUs.
- Efficient 4-bit model support.
- Practical for iterative RL experimentation.

The training loop:

1. Reset batched environments.
2. Build prompts from current observations.
3. Generate model completions.
4. Parse action JSON/text to executable actions.
5. Step environments and collect trajectories.
6. Compute GRPO loss and update model.

### 8.2 Fallback: Standard GRPO

`train_grpo.py` provides a compatible fallback without Unsloth-specific acceleration.

### 8.3 Wrapper Role

`server/grpo_env_wrapper.py` bridges free-form model output and structured environment actions.

It supports:

- JSON extraction from completions.
- Function-like action parsing.
- Safe fallback to `hold` for malformed output.

## 9. Evaluation and Diagnostics

### 9.1 Baseline Evaluation

`eval.py` runs a greedy baseline policy for reference performance.

### 9.2 Episode Diagnostics

`diagnose.py` helps inspect:

- Reward behavior
- Action distributions
- Failure/fatality patterns
- Event influence

### 9.3 Monitoring Tools

`scripts/monitor_training.py` and launch scripts help track remote or long-running training jobs.

## 10. Tests and Quality Assurance

Automated tests cover:

- Environment lifecycle behavior
- Call generation and verification paths
- Unit state machine transitions
- City graph consistency and pathing
- Hospital/event interactions
- End-to-end environment connection flow

Run tests:

```bash
python -m pytest tests/
```

## 11. Setup and Installation

### 11.1 Core Environment

```bash
pip install -e .
```

### 11.2 Development/Test Dependencies

```bash
pip install -e .[dev]
```

### 11.3 Training Dependencies

Install required training packages (project-specific choice may vary by hardware):

```bash
pip install unsloth transformers torch trl accelerate datasets bitsandbytes peft
```

Note: For Unsloth optimization, import order matters in training scripts (`import unsloth` before Transformers).

## 12. Common Run Commands

Quick validation:

```bash
python -m pytest tests/
python eval.py
```

Interactive/demo workflows:

```bash
python demo.py --episodes 3 --difficulty learning --delay 0.3
python diagnose.py --episodes 10 --agent greedy
```

Primary training example:

```bash
python train_unsloth_grpo.py \
	--model unsloth/Qwen3-4B-Thinking-2507-bnb-4bit \
	--episodes 200 \
	--batch-size 8 \
	--curriculum \
	--output-dir ./outputs/unsloth_grpo
```

Fallback training example:

```bash
python train_grpo.py \
	--model Qwen/Qwen3.5-2B \
	--episodes 200 \
	--batch-size 8
```

Run environment server:

```bash
python -m server.app --port 8000
```

## 13. India-Relevant Positioning (Pitch Support)

Although the framework is general-purpose, the current scenario maps well to India-relevant emergency operations:

- High demand and finite fleet constraints.
- Noisy real-time information channels.
- Congestion and event-driven traffic shocks.
- Hospital diversion/capacity volatility.

This makes DispatchR useful as a simulation-first platform for testing dispatch strategies safely before real-world deployment.

## 14. Extensibility

The same architecture can be adapted to other domains:

- Fire and disaster response dispatch.
- Police/patrol allocation.
- Utility outage repair fleets.
- Logistics and service routing under uncertainty.

Typical adaptation points:

- Action schema in `models.py`.
- Domain generators/schedulers in `server/`.
- Reward function in `server/reward.py`.
- Baselines and diagnostics.

## 15. Operational Tips

1. Start with deterministic seeds and smaller episode counts for debugging.
2. Validate action parsing robustness before long training runs.
3. Track invalid-action and hold-rate metrics to detect policy collapse.
4. Keep curriculum progression monitored; overly aggressive escalation can destabilize learning.
5. Log trajectories frequently for replay and postmortem analysis.

## 16. Troubleshooting

### Issue: Model outputs malformed actions

- Verify prompt constraints and final-line JSON expectations.
- Use wrapper fallback behavior and log raw completions.

### Issue: Training appears stalled

- Check reward trend, invalid action rate, and exploration settings.
- Reduce max completion length and batch size to isolate instability.

### Issue: OOM on GPU

- Use smaller model or 4-bit variants.
- Reduce batch/micro-batch sizes.
- Lower max completion length.

### Issue: Environment behavior seems unrealistic

- Review constants and event probabilities.
- Run diagnostics and inspect ground-truth output for hidden-state effects.

## 17. Suggested Read Order for New Contributors

1. `README.md`
2. `models.py`
3. `server/dispatcher_environment.py`
4. `server/call_generator.py` and `server/unit_model.py`
5. `server/reward.py`
6. `train_unsloth_grpo.py`
7. `tests/`

## 18. Summary

DispatchR is a robust, simulation-driven decision intelligence project:

- It captures uncertainty-rich dispatch operations.
- It uses RL-ready environment mechanics and reward design.
- It supports efficient training with Unsloth and fallback compatibility.
- It is test-backed, extensible, and practical for both research and real-world policy prototyping.
