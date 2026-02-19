<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Thompson Sampling Router Parameters

This document describes all configurable parameters for the `WorkloadAwareRouter` in `router.py`.

## Configuration Methods

Parameters can be set via:

1. **YAML Config File** (`config.yaml`) - All 31 parameters
2. **CLI Flags** - 5 flags for common operations:
   - `--config` - Path to YAML config file
   - `--affinity-base` - Primary stickiness control
   - `--temp-base` - Primary exploration control  
   - `--lints-v` - Exploration variance
   - `--override` - Override any config value (repeatable)

**Precedence:** CLI flags override config file values.

## Usage Examples

```bash
# Use config file only
python router.py --config config.yaml

# Override specific values
python router.py --config config.yaml --affinity-base 0.5 --temp-base 1.5

# Override nested values
python router.py --config config.yaml --override load_balancing.gpu_penalty_weight=2.0

# Multiple overrides
python router.py --config config.yaml \
  --override switching_cost.base=0.3 \
  --override feedback.timeout_seconds=60
```

---

## Parameter Reference

### Infrastructure

| Parameter | Config Path | Default | Type | Description |
|-----------|-------------|---------|------|-------------|
| Block Size | `infrastructure.block_size` | 64 | int | KV cache block size for overlap computation |
| Router Type | `infrastructure.router_type` | "kv" | str | Router mode: "kv" (KV-aware) or "kv_load" (load-based only) |
| Min Workers | `infrastructure.min_workers` | 1 | int | Minimum workers required before routing starts |

### Affinity (Stickiness)

Controls how strongly the router prefers keeping requests on the same worker for KV cache reuse.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Base | `affinity.base` | `--affinity-base` | 0.30 | float | Base bonus when staying on same worker. Higher = more sticky. |
| Reuse Weight | `affinity.reuse_weight` | `--override` | 0.15 | float | Additional bonus per remaining request in session |
| IAT Weight | `affinity.iat_weight` | `--override` | 0.20 | float | Bonus scaling based on inter-arrival time hint |
| Sticky Load Floor | `affinity.sticky_load_floor` | `--override` | 0.70 | float | Minimum load modifier for sticky decisions (prevents load from overriding stickiness) |

**Tuning Guide:**
- **High affinity (0.4-0.6):** Prioritize KV cache hits, good for multi-turn conversations
- **Low affinity (0.1-0.2):** Prioritize load balancing, good for independent requests

### Exploration (Temperature)

Controls the explore vs exploit tradeoff in worker selection.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Base TS Weight | `exploration.base_ts_weight` | `--override` | 0.10 | float | Weight for Thompson Sampling exploration term |
| Temp Base | `exploration.temperature.base` | `--temp-base` | 1.0 | float | Base softmax temperature |
| Temp Min | `exploration.temperature.min` | `--override` | 0.15 | float | Minimum temperature (more greedy selection) |
| Temp Max | `exploration.temperature.max` | `--override` | 2.0 | float | Maximum temperature (more random selection) |

**Tuning Guide:**
- **Low temperature (0.2-0.5):** Greedy, always pick best-scored worker
- **High temperature (1.5-2.0):** More exploration, useful early or with changing workloads

### Switching Cost

Penalty applied when moving a prefix session to a different worker.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Base | `switching_cost.base` | `--override` | 0.20 | float | Base penalty for switching workers |
| Reuse Penalty | `switching_cost.reuse_penalty` | `--override` | 0.08 | float | Additional penalty per remaining request in session |
| IAT Penalty | `switching_cost.iat_penalty` | `--override` | 0.05 | float | Additional penalty based on inter-arrival time |

**Tuning Guide:**
- **High switching cost (0.3-0.5):** Strongly discourage worker changes mid-session
- **Low switching cost (0.05-0.1):** Allow flexible rebalancing

### Load Balancing

Controls how much to penalize workers with high load.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Queue Penalty | `load_balancing.queue_penalty_weight` | `--override` | 0.50 | float | Weight for pending request queue depth |
| GPU Penalty | `load_balancing.gpu_penalty_weight` | `--override` | 1.00 | float | Weight for GPU KV cache memory usage |
| Outstanding Work | `load_balancing.outstanding_work_weight` | `--override` | 0.45 | float | Weight for outstanding work (expected future load) |
| Job-GPU Coupling | `load_balancing.job_gpu_coupling_weight` | `--override` | 0.40 | float | How much job cost amplifies GPU load penalty |
| Job-Queue Coupling | `load_balancing.job_queue_coupling_weight` | `--override` | 0.20 | float | How much job cost amplifies queue penalty |

**Tuning Guide:**
- **High GPU penalty (1.5-2.0):** Aggressively avoid memory-constrained workers
- **High queue penalty (0.8-1.0):** Prioritize low-latency routing

### Prefill Cost Model

How input sequence length (ISL) contributes to job cost estimation.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Token Scale | `prefill.token_scale` | `--override` | 1024.0 | float | Normalization denominator for token count |
| Weight | `prefill.weight` | `--override` | 1.0 | float | Weight of prefill cost in total job cost |

### LinTS Learner

Parameters controlling the Linear Thompson Sampling algorithm.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Lambda | `lints.lambda` | `--override` | 1.0 | float | Ridge regression regularization strength |
| V | `lints.v` | `--lints-v` | 0.25 | float | Exploration variance in posterior sampling |
| Forget Rate | `lints.forget_rate` | `--override` | 0.995 | float | Exponential decay for old observations |

**Tuning Guide:**
- **High V (0.4-0.6):** More exploration, keeps trying alternatives
- **Low V (0.05-0.15):** More exploitation, trusts learned model
- **High forget rate (0.999):** Long memory, slow adaptation
- **Low forget rate (0.95):** Short memory, fast adaptation to changes

### Feedback Handling

Controls the delayed reward mechanism.

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Timeout Seconds | `feedback.timeout_seconds` | `--override` | 120.0 | float | Max wait time for feedback before applying timeout penalty |
| Sweep Interval | `feedback.sweep_interval_seconds` | `--override` | 5.0 | float | How often to check for timed-out decisions |
| Timeout Reward | `feedback.timeout_reward` | `--override` | 0.0 | float | Reward for timed-out requests (0.0 = treat as failure) |
| Latency EMA Alpha | `feedback.latency_ema_alpha` | `--override` | 0.2 | float | Smoothing factor for latency baseline EMA |

### Debug

| Parameter | Config Path | CLI Flag | Default | Type | Description |
|-----------|-------------|----------|---------|------|-------------|
| Traces Enabled | `debug.traces_enabled` | `--override` | false | bool | Enable detailed decision trace logging |
| Trace Dir | `debug.trace_dir` | `--override` | /tmp/dynamo_router_traces | str | Directory for trace output files |
| Buffer Size | `debug.buffer_size` | `--override` | 2000 | int | In-memory trace buffer size |

---

## Feature Vector (LinTS Input)

The router uses a 9-dimensional feature vector as input to the LinTS learner:

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0 | Bias | Constant | Always 1.0 (intercept term) |
| 1 | Inverse Load | Computed | 1/(1 + gpu_penalty + queue_penalty) |
| 2 | Overlap | KV Indexer | KV cache overlap score [0, 1] |
| 3 | Affinity | State | 1 if same worker as last request, else 0 |
| 4 | Outstanding | State | tanh(0.1 × outstanding_work) |
| 5 | Decode Cost | OSL Hint | decode_cost / 3.0 |
| 6 | Prefill Cost | Computed | tanh(prefill_cost) |
| 7 | IAT Factor | IAT Hint | iat_factor / 1.5 |
| 8 | Reuse Budget | Hint | tanh(0.25 × reuse_budget) |

---

## Learned vs Fixed Parameters

| Category | Updated At | Examples |
|----------|-----------|----------|
| **Learned (runtime)** | Every request | `linA`, `linb` matrices, Beta(α,β) bandits, latency EMAs |
| **Fixed (startup)** | Never | All 31 config parameters above |

The config parameters are **hyperparameters** that control *how* the router learns, not *what* it learns.

