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


<!-- path-check-skip-begin -->

# Finetuning Harness

:::{warning}
**Experimental Feature**: The Finetuning Harness is experimental and may change in future releases. Future versions may introduce breaking changes without notice.
:::

The NeMo Agent Toolkit provides a powerful finetuning harness designed for **in-situ reinforcement learning** of agentic [LLM](../../build-workflows/llms/index.md) [workflows](../../build-workflows/about-building-workflows.md). This enables iterative improvement of [agents](../../components/agents/index.md) through experience, allowing models to learn from their interactions with environments, [tools](../../build-workflows/functions-and-function-groups/functions.md#agents-and-tools), and users.

## Overview

The finetuning harness is built on four foundational principles:

| Principle | Description |
|-----------|-------------|
| **Decoupled Architecture** | Training logic is separated from backends, allowing you to use any RL framework (OpenPipe ART, NeMo Aligner, custom implementations) |
| **In-Situ Training** | Train agents with the same workflow you run in production, without moving to a different development environment.|
| **Flexible Targeting** | Finetune specific [functions](../../build-workflows/functions-and-function-groups/functions.md) or entire workflows, enabling targeted improvements in complex agentic systems. |
| **Composable Components** | Three pluggable components (TrajectoryBuilder, TrainerAdapter, Trainer) can be mixed, matched, and customized |

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                              Trainer                                   │
│  (Orchestrates the finetuning loop across epochs)                      │
│                                                                        │
│  ┌───────────────────────┐         ┌───────────────────────────┐       │
│  │  TrajectoryBuilder    │         │    TrainerAdapter         │       │
│  │                       │         │                           │       │
│  │  - Runs evaluations   │ ──────► │  - Validates trajectories │       │
│  │  - Collects episodes  │         │  - Submits to backend     │       │
│  │  - Computes rewards   │         │  - Monitors training      │       │
│  │  - Groups trajectories│         │  - Reports status         │       │
│  └───────────────────────┘         └───────────────────────────┘       │
└────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   Remote Training       │
                            │      Backend            │
                            └─────────────────────────┘
```

## Documentation

| Guide | Description |
|-------|-------------|
| [Concepts](concepts.md) | Core concepts, RL fundamentals, curriculum learning, and architecture details |
| [Extending](../../extend/custom-components/finetuning.md) | How to implement custom TrajectoryBuilders, TrainerAdapters, and Trainers |
| [OpenPipe ART](rl_with_openpipe.md) | Using the OpenPipe ART backend for GRPO training |

## Supported Backends

| Backend | Plugin Package | Description |
|---------|----------------|-------------|
| OpenPipe ART | `nvidia-nat-openpipe-art` | GRPO-based training with vLLM and TorchTune |

## Key Features

- **Curriculum Learning**: Progressively introduce harder examples during training
- **Multi-Generation Trajectories**: Collect multiple responses per example for GRPO optimization
- **Validation Monitoring**: Periodic evaluation on held-out data to track generalization
- **Progress Visualization**: Automatic reward plots and metrics logging
- **Flexible Targeting**: Train specific functions or models in complex workflows

## Requirements

- Training backend (e.g., OpenPipe ART server with GPU)
- LLM inference endpoint with log probability support
- Training dataset in JSON/JSONL format
- Custom evaluator for computing rewards

<!-- path-check-skip-end -->

```{toctree}
:hidden:
:caption: Finetuning

Concepts <./concepts.md>
OpenPipe ART <./rl_with_openpipe.md>
DPO With NeMo Customizer <./dpo_with_nemo_customizer.md>
```
