<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# OpenPipe ART Integration

This guide covers the integration between NAT's finetuning harness and [OpenPipe ART](https://art.openpipe.ai/) (Agent Reinforcement Trainer), an open-source framework for teaching LLMs through reinforcement learning.

## About OpenPipe ART

OpenPipe ART is designed to improve agent **performance and reliability through experience**. It provides:

- **GRPO Training**: Uses Group Relative Policy Optimization, which compares multiple responses to the same prompt rather than requiring a separate value function
- **Async Client-Server Architecture**: Separates inference from training, allowing you to run inference anywhere while training happens on GPU infrastructure
- **Easy Integration**: Designed to work with existing LLM applications with minimal code changes
- **Built-in Observability**: Integrations with Weights & Biases, Langfuse, and OpenPipe for monitoring and debugging

### When to Use ART

ART is well-suited for scenarios where:

- You want to improve agent reliability on specific tasks
- You have a way to score agent performance (even if you don't have "correct" answers)
- You're working with agentic workflows that make decisions or take actions
- You want to iterate quickly with online training

### ART Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Your Application                              │
│                                                                         │
│   ┌─────────────────────┐                                               │
│   │   NAT Workflow      │ ◄──── Uses model for inference                │
│   └─────────────────────┘                                               │
│            │                                                            │
│            │ Trajectories                                               │
│            ▼                                                            │
│   ┌─────────────────────┐         ┌─────────────────────────────────┐   │
│   │ ARTTrajectoryBuilder│────────►│      ART Backend Server         │   │
│   │  ARTTrainerAdapter  │         │                                 │   │
│   └─────────────────────┘         │  ┌─────────────────────────────┐│   │
│            │                      │  │  vLLM Inference Engine      ││   │
│            │ Training request     │  │  (serves updated weights)   ││   │
│            │                      │  └─────────────────────────────┘│   │
│            └─────────────────────►│  ┌─────────────────────────────┐│   │
│                                   │  │  GRPO Trainer (TorchTune)   ││   │
│                                   │  │  (updates model weights)    ││   │
│                                   │  └─────────────────────────────┘│   │
│                                   └─────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

The ART backend runs on GPU infrastructure and provides:
- **vLLM Inference Engine**: Serves the model for inference with log probability support
- **GRPO Trainer**: Performs weight updates based on submitted trajectories

NAT connects to this backend through the `ARTTrainerAdapter`, which handles the protocol for submitting trajectories and monitoring training.

## Installation

Install the OpenPipe ART plugin package:

```bash
pip install nvidia-nat-openpipe-art
```

This provides:
- `openpipe_art_traj_builder`: The trajectory builder implementation
- `openpipe_art_trainer_adapter`: The trainer adapter for ART
- `openpipe_art_trainer`: The trainer orchestrator

You'll also need to set up an ART backend server. See the [ART documentation](https://art.openpipe.ai/getting-started/about) for server setup instructions.

## Configuration

### Basic Configuration

```yaml
llms:
  training_llm:
    _type: openai
    model_name: Qwen/Qwen2.5-3B-Instruct
    base_url: http://localhost:8000/v1  # ART inference endpoint
    api_key: default
    temperature: 0.4

workflow:
  _type: my_workflow
  llm: training_llm

eval:
  general:
    max_concurrency: 16
    output_dir: .tmp/nat/finetuning/eval
    dataset:
      _type: json
      file_path: data/training_data.json

  evaluators:
    my_reward:
      _type: my_custom_evaluator

trajectory_builders:
  art_builder:
    _type: openpipe_art_traj_builder
    num_generations: 2

trainer_adapters:
  art_adapter:
    _type: openpipe_art_trainer_adapter
    backend:
      ip: "localhost"
      port: 7623
      name: "my_training_run"
      project: "my_project"
      base_model: "Qwen/Qwen2.5-3B-Instruct"
      api_key: "default"
    training:
      learning_rate: 1e-6

trainers:
  art_trainer:
    _type: openpipe_art_trainer

finetuning:
  enabled: true
  trainer: art_trainer
  trajectory_builder: art_builder
  trainer_adapter: art_adapter
  reward_function:
    name: my_reward
  num_epochs: 20
  output_dir: .tmp/nat/finetuning/output
```

### Configuration Reference

#### Trajectory Builder Configuration

```yaml
trajectory_builders:
  art_builder:
    _type: openpipe_art_traj_builder
    num_generations: 2  # Trajectories per example
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_generations` | `int` | `2` | Number of trajectory generations per example. More generations provide better GRPO signal but increase computation time. |

#### Trainer Adapter Configuration

```yaml
trainer_adapters:
  art_adapter:
    _type: openpipe_art_trainer_adapter
    backend:
      ip: "0.0.0.0"
      port: 7623
      name: "training_run_name"
      project: "project_name"
      base_model: "Qwen/Qwen2.5-3B-Instruct"
      api_key: "default"
      delete_old_checkpoints: false
      init_args:
        max_seq_length: 8192
      engine_args:
        gpu_memory_utilization: 0.9
        tensor_parallel_size: 1
    training:
      learning_rate: 1e-6
      beta: 0.0
```

**Backend Configuration**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ip` | `str` | - | IP address of the ART backend server |
| `port` | `int` | - | Port of the ART backend server |
| `name` | `str` | `"trainer_run"` | Name for this training run |
| `project` | `str` | `"trainer_project"` | Project name for organization |
| `base_model` | `str` | `"Qwen/Qwen2.5-7B-Instruct"` | Base model being trained (must match server) |
| `api_key` | `str` | `"default"` | API key for authentication |
| `delete_old_checkpoints` | `bool` | `false` | Delete old checkpoints before training |

**Model Initialization Arguments (`init_args`)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_seq_length` | `int` | - | Maximum sequence length for the model |

**vLLM Engine Arguments (`engine_args`)**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `gpu_memory_utilization` | `float` | - | Fraction of GPU memory to use (0.0-1.0) |
| `tensor_parallel_size` | `int` | - | Number of GPUs for tensor parallelism |

**Training Arguments**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `learning_rate` | `float` | `5e-5` | Learning rate for GRPO updates |
| `beta` | `float` | `0.0` | KL penalty coefficient |

#### Trainer Configuration

```yaml
trainers:
  art_trainer:
    _type: openpipe_art_trainer
```

The trainer has no additional configuration options; it uses the shared `finetuning` configuration.

## How It Works

### ARTTrajectoryBuilder

The `ARTTrajectoryBuilder` collects training trajectories through NAT's evaluation system:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ARTTrajectoryBuilder Flow                           │
│                                                                         │
│  start_run()                                                            │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Launch N parallel evaluation runs (num_generations)               │ │
│  │                                                                    │ │
│  │  Each run:                                                         │ │
│  │    1. Loads the training dataset                                   │ │
│  │    2. Runs the workflow on each example                            │ │
│  │    3. Captures intermediate steps (with logprobs from LLM calls)   │ │
│  │    4. Computes reward using configured evaluator                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                  │
│      ▼                                                                  │
│  finalize()                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Wait for all evaluation runs to complete                          │ │
│  │                                                                    │ │
│  │  For each result:                                                  │ │
│  │    1. Extract reward from evaluator output                         │ │
│  │    2. Filter intermediate steps to target functions                │ │
│  │    3. Parse steps into OpenAI message format                       │ │
│  │    4. Validate assistant messages have logprobs                    │ │
│  │    5. Group trajectories by example ID                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                  │
│      ▼                                                                  │
│  Return TrajectoryCollection                                            │
│  (grouped by example for GRPO)                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Implementation Details**:

1. **Parallel Generation**: Multiple evaluation runs execute concurrently using `asyncio.create_task()`. This generates diverse trajectories for the same inputs.

2. **Log Probability Extraction**: The builder parses intermediate steps to extract log probabilities from LLM responses. Messages without logprobs are skipped since they can't be used for training.

3. **Target Function Filtering**: Only steps from functions listed in `finetuning.target_functions` are included. This lets you focus training on specific parts of complex workflows.

4. **Grouping for GRPO**: Trajectories are organized as `list[list[Trajectory]]` where each inner list contains all generations for a single example. This structure enables group-relative policy optimization.

### ARTTrainerAdapter

The `ARTTrainerAdapter` converts NAT trajectories to ART's format and manages training:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ARTTrainerAdapter Flow                              │
│                                                                         │
│  initialize()                                                           │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  1. Create ART Backend client                                      │ │
│  │  2. Create TrainableModel with configuration                       │ │
│  │  3. Register model with backend                                    │ │
│  │  4. Verify backend health                                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                  │
│      ▼                                                                  │
│  submit(trajectories)                                                   │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  1. Validate episode ordering                                      │ │
│  │     - First message: user or system                                │ │
│  │     - No consecutive assistant messages                            │ │
│  │                                                                    │ │
│  │  2. Convert to ART TrajectoryGroup format                          │ │
│  │     - EpisodeItem → dict or Choice                                 │ │
│  │     - Include logprobs in Choice objects                           │ │
│  │                                                                    │ │
│  │  3. Submit via model.train() (async)                               │ │
│  │                                                                    │ │
│  │  4. Return TrainingJobRef for tracking                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                  │
│      ▼                                                                  │
│  wait_until_complete()                                                  │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Poll task status until done                                       │ │
│  │  Return final TrainingJobStatus                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Implementation Details**:

1. **ART Client Management**: The adapter maintains an `art.Backend` client and `art.TrainableModel` instance that persist across epochs.

2. **Trajectory Conversion**: NAT's `Trajectory` objects are converted to ART's `art.Trajectory` format:
   ```python
   # NAT format
   EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="...", logprobs=...)

   # Converted to ART format
   Choice(index=0, logprobs=..., message={"role": "assistant", "content": "..."}, finish_reason="stop")
   ```

3. **Message Validation**: The adapter validates that conversations follow expected patterns (user/system first, no consecutive assistant messages).

4. **Async Training**: Training is submitted as an async task, allowing the trainer to monitor progress without blocking.

### ARTTrainer

The `ARTTrainer` orchestrates the complete training loop:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARTTrainer Flow                                  │
│                                                                         │
│  initialize()                                                           │
│      │                                                                  │
│      ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  1. Generate unique run ID                                         │ │
│  │  2. Initialize trajectory builder                                  │ │
│  │  3. Initialize trainer adapter                                     │ │
│  │  4. Set up curriculum learning state                               │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│      │                                                                  │
│      ▼                                                                  │
│  run(num_epochs)                                                        │
│      │                                                                  │
│  for epoch in range(num_epochs):                                        │
│      │                                                                  │
│      ├─── Validation (if interval reached) ─────────────────────────┐   │
│      │                                                              │   │
│      │    ┌───────────────────────────────────────────────────────┐ │   │
│      │    │  Run evaluation on validation dataset                 │ │   │
│      │    │  Record metrics (avg_reward, etc.)                    │ │   │
│      │    │  Store in validation history                          │ │   │
│      │    └───────────────────────────────────────────────────────┘ │   │
│      │                                                              │   │
│      ◄──────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ├─── run_epoch() ──────────────────────────────────────────────┐   │
│      │                                                              │   │
│      │    ┌───────────────────────────────────────────────────────┐ │   │
│      │    │  1. Start trajectory collection                       │ │   │
│      │    │  2. Finalize and compute metrics                      │ │   │
│      │    │  3. Apply curriculum learning (filter groups)         │ │   │
│      │    │  4. Submit to trainer adapter                         │ │   │
│      │    │  5. Log progress and generate plots                   │ │   │
│      │    └───────────────────────────────────────────────────────┘ │   │
│      │                                                              │   │
│      ◄──────────────────────────────────────────────────────────────┘   │
│      │                                                                  │
│      ├─── Wait for training to complete                                 │
│      │                                                                  │
│      └─── Check status, break on failure                                │
│                                                                         │
│  Return list of TrainingJobStatus                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Implementation Details**:

1. **Curriculum Learning**: The trainer implements curriculum learning to progressively include harder examples:
   - Groups trajectories by average reward
   - Filters out groups with insufficient variance (no learning signal)
   - Starts with easiest fraction, expands at intervals

2. **Validation**: Optionally runs evaluation on a separate validation dataset to monitor generalization.

3. **Progress Visualization**: Generates reward plots (`reward_plot.png`) showing training and validation reward progression.

4. **Metrics Logging**: Writes detailed metrics to JSONL files for analysis:
   - `training_metrics.jsonl`: Per-epoch metrics
   - `reward_history.json`: Reward progression
   - `curriculum_state.json`: Curriculum learning state

## Running Finetuning

### Prerequisites

1. **ART Backend Server**: You need a running ART server with your model loaded. See [ART documentation](https://art.openpipe.ai/) for setup.

2. **LLM with Logprobs**: Your LLM must return log probabilities. For vLLM, use the `--enable-log-probs` flag.

3. **Training Dataset**: A JSON/JSONL dataset with your training examples.

4. **Reward Function**: An evaluator that can score workflow outputs.


### Running Training

You must have OpenPipe ART plugin installed (`nvidia-nat-openpipe-art`), and an OpenPipe ART server running 
and configured to accept training jobs.

```bash
# Basic training
nat finetune --config_file=configs/finetune.yml

# With validation
nat finetune --config_file=configs/finetune.yml \
    --validation_dataset=data/val.json \
    --validation_interval=5

# Override epochs
nat finetune --config_file=configs/finetune.yml \
    -o finetuning.num_epochs 50
```

### Monitoring Progress

During training, check:

1. **Console Output**: Shows epoch progress, reward statistics, trajectory counts

2. **Metrics Files**: In your `output_dir`:
   - `training_metrics.jsonl`: Detailed per-epoch metrics
   - `reward_plot.png`: Visual reward progression
   - `reward_history.json`: Raw reward data

3. **ART Server Logs**: Training progress from the ART side

Example console output:
```
INFO - Starting epoch 1 for run art_run_a1b2c3d4
INFO - Starting 2 evaluation runs for run_id: art_run_a1b2c3d4
INFO - Built 100 trajectories across 50 examples for run_id: art_run_a1b2c3d4
INFO - Submitted 100 trajectories in 50 groups for training
INFO - Epoch 1 progress logged - Avg Reward: 0.4523, Trajectories: 100
INFO - Training art_run_a1b2c3d4 completed successfully.
INFO - Completed epoch 1/20
```

## Advanced Configuration

### Multi-GPU Training

For larger models, configure tensor parallelism:

```yaml
trainer_adapters:
  art_adapter:
    _type: openpipe_art_trainer_adapter
    backend:
      engine_args:
        tensor_parallel_size: 2  # Use 2 GPUs
        gpu_memory_utilization: 0.85
```

### Memory Optimization

If you encounter OOM errors:

```yaml
trainer_adapters:
  art_adapter:
    _type: openpipe_art_trainer_adapter
    backend:
      init_args:
        max_seq_length: 4096  # Reduce sequence length
      engine_args:
        gpu_memory_utilization: 0.7  # Leave more headroom
```

### Curriculum Learning

Enable curriculum learning to improve training stability:

```yaml
finetuning:
  curriculum_learning:
    enabled: true
    initial_percentile: 0.3      # Start with easiest 30%
    increment_percentile: 0.2     # Add 20% each expansion
    expansion_interval: 5         # Expand every 5 epochs
    min_reward_diff: 0.1         # Filter no-variance groups
    sort_ascending: false         # Easy-to-hard
```

### Targeting Specific Functions

For multi-component workflows, target specific functions:

```yaml
finetuning:
  target_functions:
    - my_agent_function
    - tool_calling_function
  target_model: training_llm  # Only include steps from this model
```

## Troubleshooting

### Connection Issues

**"Failed to connect to ART backend"**

1. Verify the server is running:
   ```bash
   curl http://localhost:7623/health
   ```

2. Check IP and port in configuration

3. Verify network connectivity (firewalls, etc.)

### Missing Log Probabilities

**"No valid assistant messages with logprobs"**

1. Ensure your LLM provider returns logprobs
2. For vLLM: verify `--enable-log-probs` flag
3. Check your NAT LLM configuration

### Out of Memory

**"CUDA out of memory"**

1. Reduce `gpu_memory_utilization`
2. Reduce `max_seq_length`
3. Reduce `num_generations` (fewer parallel trajectories)
4. Increase `tensor_parallel_size` (distribute across GPUs)

### No Trajectories Collected

**"No trajectories collected for epoch"**

1. Check `target_functions` matches your workflow
2. Verify workflow produces intermediate steps
3. Check evaluator is returning rewards
4. Look for errors in evaluation logs

### Training Not Improving

**Rewards not increasing**

1. Increase `num_generations` for better GRPO signal
2. Try curriculum learning to focus on learnable examples
3. Adjust learning rate
4. Verify reward function is well-calibrated
5. Check for sufficient variance in trajectory groups

## Examples

The `examples/finetuning/rl_with_openpipe_art` directory contains a complete working example demonstrating:

- Custom workflow with intermediate step tracking
- Custom reward evaluator with reward shaping
- Full configuration for ART integration
- Training and evaluation datasets

See the example's README for detailed instructions.

## See Also

- [Finetuning Concepts](concepts.md) - Core concepts and RL fundamentals
- [Extending the Finetuning Harness](extending.md) - Creating custom components
- [OpenPipe ART Documentation](https://art.openpipe.ai/) - Official ART documentation
- [Custom Evaluators](../extend/custom-evaluator.md) - Creating reward functions
