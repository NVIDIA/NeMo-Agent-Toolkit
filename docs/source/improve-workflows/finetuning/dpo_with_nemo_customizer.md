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

# DPO with NeMo Customizer

This guide covers Direct Preference Optimization (DPO) training using the NeMo Agent toolkit finetuning harness integrated with [NVIDIA NeMo Customizer](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/index.html). 
This integration enables preference-based finetuning of large language models using NVIDIA's enterprise-grade training infrastructure.

## Understanding DPO

### What is Direct Preference Optimization?

Direct Preference Optimization (DPO) is a reinforcement learning technique that trains language models to prefer certain responses over others, without requiring a separate reward model. Unlike traditional RLHF (Reinforcement Learning from Human Feedback), which requires training a reward model and then using PPO to optimize against it, DPO directly optimizes the policy using preference pairs.

### The DPO Objective

DPO works by optimizing the following objective:

```
L_DPO(π_θ; π_ref) = -E[(x, y_w, y_l)] [log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x)) - β · (log π_θ(y_l|x) - log π_ref(y_l|x)))]
```

Where:
- `π_θ` is the policy being trained
- `π_ref` is the reference policy (frozen copy of the initial model)
- `x` is the prompt
- `y_w` is the "chosen" (preferred) response
- `y_l` is the "rejected" (non-preferred) response
- `β` is a temperature parameter controlling deviation from the reference policy
- `σ` is the sigmoid function

In simpler terms: DPO increases the probability of chosen responses while decreasing the probability of rejected responses, with a KL penalty to prevent the model from deviating too far from its original behavior.

### Why DPO?

**Advantages over traditional RLHF:**

1. **Simpler Pipeline**: No need to train a separate reward model
2. **More Stable Training**: Avoids the instabilities of PPO optimization
3. **Computationally Efficient**: Single-stage training process
4. **Direct Optimization**: Directly optimizes preference likelihood

**When to use DPO:**

- You have paired preference data (chosen vs rejected responses)
- You want to align model outputs with specific quality criteria
- You're training agents where you can score different action choices
- You want to improve response quality without explicit reward modeling

### Preference Pairs from Test-Time Compute

The NeMo Agent toolkit DPO integration uses Test-Time Compute (TTC) to generate preference pairs automatically. During workflow execution:

1. **Multiple Candidates Generated**: For each decision point, the workflow generates multiple candidate responses
2. **Candidates Scored**: Each candidate is evaluated using a scoring function
3. **Pairs Created**: Higher-scored candidates become "chosen", lower-scored become "rejected"

This approach enables automated preference data collection without manual labeling.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DPO Training Pipeline                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Data Collection Phase                              ││
│  │                                                                         ││
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐   ││
│  │  │   Dataset    │───►│   Workflow   │───►│  TTC Move Selector       │   ││
│  │  │   (inputs)   │    │  Execution   │    │  (generates candidates)  │   ││
│  │  └──────────────┘    └──────────────┘    └──────────────────────────┘   ││
│  │                                                   │                     ││
│  │                                                   ▼                     ││
│  │                                          ┌──────────────────────────┐   ││
│  │                                          │   Score Candidates       │   ││
│  │                                          │   (reward function)      │   ││
│  │                                          └──────────────────────────┘   ││
│  │                                                   │                     ││
│  │                                                   ▼                     ││
│  │  ┌──────────────────────────────────────────────────────────────────┐   ││
│  │  │                  DPO Trajectory Builder                          │   ││
│  │  │                                                                  │   ││
│  │  │  • Collects TTC_END intermediate steps with TTCEventData         │   ││
│  │  │  • Groups candidates by turn_id                                  │   ││
│  │  │  • Generates preference pairs (chosen vs rejected)               │   ││
│  │  │  • Builds Trajectory objects with DPOItem episodes               │   ││
│  │  └──────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      Training Submission Phase                          ││
│  │                                                                         ││
│  │  ┌──────────────────────────────────────────────────────────────────┐   ││
│  │  │                  NeMo Customizer Trainer Adapter                 │   ││
│  │  │                                                                  │   ││
│  │  │  1. Convert trajectories to JSONL format                         │   ││
│  │  │  2. Upload dataset to NeMo Datastore (via HuggingFace Hub API)   │   ││
│  │  │  3. Submit customization job to NeMo Customizer                  │   ││
│  │  │  4. Monitor job progress until completion                        │   ││
│  │  │  5. Optionally deploy trained model                              │   ││
│  │  └──────────────────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    NeMo Customizer Backend                              ││
│  │                                                                         ││
│  │  ┌─────────────────────┐  ┌─────────────────────┐                       ││
│  │  │   Entity Store      │  │   Datastore         │                       ││
│  │  │   (job management)  │  │   (dataset storage) │                       ││
│  │  └─────────────────────┘  └─────────────────────┘                       ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐    ││
│  │  │                    Training Infrastructure                      │    ││
│  │  │                                                                 │    ││
│  │  │  • DPO loss computation with reference model                    │    ││
│  │  │  • LoRA or full-weight finetuning                               │    ││
│  │  │  • Multi-GPU distributed training                               │    ││
│  │  └─────────────────────────────────────────────────────────────────┘    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                    │                                        │
│                                    ▼                                        │
│                         ┌──────────────────┐                                │
│                         │  Trained Model   │                                │
│                         │  (optional NIM   │                                │
│                         │   deployment)    │                                │
│                         └──────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Installation

Install the NeMo Customizer plugin package:

```bash
pip install nvidia-nat-nemo-customizer
```

This provides:
- `dpo_traj_builder`: DPO trajectory builder for collecting preference pairs
- `nemo_customizer_trainer_adapter`: Adapter for submitting jobs to NeMo Customizer
- `nemo_customizer_trainer`: Trainer orchestrator for the DPO workflow

### Prerequisites

1. **NeMo Microservices Platform (NMP)**: Access to a deployed NeMo Customizer instance
2. **Entity Store**: For managing datasets, models, and jobs
3. **Datastore**: For storing training datasets (accessed via HuggingFace Hub API)

## Configuration

### Complete Configuration Example

```yaml
# LLM Configuration
llms:
  inference_llm:
    _type: openai
    model_name: meta/llama-3.1-8b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    api_key: ${NVIDIA_API_KEY}
    temperature: 0.7

# Workflow that uses TTC for candidate generation
workflow:
  _type: my_dpo_workflow
  llm: inference_llm

# Evaluation configuration
eval:
  general:
    max_concurrency: 8
    output_dir: .tmp/nat/finetuning/eval
    dataset:
      _type: json
      file_path: data/training_data.json

  evaluators:
    game_evaluator:
      _type: my_game_evaluator

# DPO Trajectory Builder
trajectory_builders:
  dpo_builder:
    _type: dpo_traj_builder
    ttc_step_name: dpo_candidate_move
    exhaustive_pairs: true
    min_score_diff: 0.05
    max_pairs_per_turn: 10
    reward_from_score_diff: true
    require_multiple_candidates: true

# NeMo Customizer Trainer Adapter
trainer_adapters:
  nemo_adapter:
    _type: nemo_customizer_trainer_adapter
    entity_host: https://nmp.example.com
    datastore_host: https://datastore.example.com
    namespace: my-dpo-project
    dataset_name: dpo-training-data
    customization_config: meta/llama-3.1-8b-instruct@v1.0.0+A100
    create_namespace_if_missing: true
    use_full_message_history: true
    hyperparameters:
      training_type: dpo
      finetuning_type: all_weights
      epochs: 3
      batch_size: 4
      learning_rate: 5e-6
      dpo:
        ref_policy_kl_penalty: 0.1
        preference_loss_weight: 1.0
        preference_average_log_probs: false
        sft_loss_weight: 0.0
    deploy_on_completion: false
    poll_interval_seconds: 30.0
    deployment_timeout_seconds: 1800.0

# NeMo Customizer Trainer
trainers:
  nemo_trainer:
    _type: nemo_customizer_trainer
    num_runs: 3
    wait_for_completion: true
    deduplicate_pairs: true
    max_pairs: 5000

# Finetuning configuration
finetuning:
  enabled: true
  trainer: nemo_trainer
  trajectory_builder: dpo_builder
  trainer_adapter: nemo_adapter
  reward_function:
    name: game_evaluator
  num_epochs: 1  # Not used for NeMo Customizer (uses num_runs instead)
  output_dir: .tmp/nat/finetuning/output
```

## Configuration Reference

### DPO Trajectory Builder Configuration

The DPO trajectory builder collects preference pairs from TTC intermediate steps.

```yaml
trajectory_builders:
  dpo_builder:
    _type: dpo_traj_builder
    ttc_step_name: dpo_candidate_move
    exhaustive_pairs: true
    min_score_diff: 0.0
    max_pairs_per_turn: null
    reward_from_score_diff: true
    require_multiple_candidates: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ttc_step_name` | `str` | `"dpo_candidate_move"` | Name of the TTC intermediate step to collect. Must match the name used in your workflow's `push_intermediate_step()` call. |
| `exhaustive_pairs` | `bool` | `true` | If `true`, generate all pairwise comparisons where `score(A) > score(B)`. If `false`, only generate best vs worst pair per turn. |
| `min_score_diff` | `float` | `0.0` | Minimum score difference required to create a preference pair. Pairs with smaller differences are filtered out. Useful for ensuring meaningful preference signal. |
| `max_pairs_per_turn` | `int \| null` | `null` | Maximum preference pairs per turn. If set, pairs are sorted by score difference (highest first) and truncated. `null` means no limit. |
| `reward_from_score_diff` | `bool` | `true` | If `true`, trajectory reward = score difference (chosen - rejected). If `false`, reward = chosen candidate's score. |
| `require_multiple_candidates` | `bool` | `true` | If `true`, skip turns with only one candidate (no preference signal possible). If `false`, include single-candidate turns. |

#### Pair Generation Modes

**Exhaustive Pairs (`exhaustive_pairs: true`)**

For candidates with scores `[A=0.9, B=0.7, C=0.5]`, generates:
- (A chosen, B rejected) - score diff: 0.2
- (A chosen, C rejected) - score diff: 0.4
- (B chosen, C rejected) - score diff: 0.2

This provides more training signal but may include weak preference pairs.

**Best vs Worst (`exhaustive_pairs: false`)**

For the same candidates, generates only:
- (A chosen, C rejected) - score diff: 0.4

This provides stronger preference signal but fewer training examples.

### NeMo Customizer Trainer Configuration

The trainer orchestrates data collection runs.

```yaml
trainers:
  nemo_trainer:
    _type: nemo_customizer_trainer
    num_runs: 3
    continue_on_collection_error: false
    deduplicate_pairs: true
    max_pairs: null
    wait_for_completion: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_runs` | `int` | `1` | Number of times to run the trajectory builder to collect data. Multiple runs increase dataset diversity by generating different trajectories for the same inputs. |
| `continue_on_collection_error` | `bool` | `false` | If `true`, continue with remaining runs if one fails. If `false`, stop immediately on first error. |
| `deduplicate_pairs` | `bool` | `true` | If `true`, remove duplicate DPO pairs based on prompt+chosen+rejected content. Useful when multiple runs may generate identical pairs. |
| `max_pairs` | `int \| null` | `null` | Maximum DPO pairs to include in training. If set, randomly samples from collected pairs. `null` means use all pairs. |
| `wait_for_completion` | `bool` | `true` | If `true`, wait for NeMo Customizer job to complete. If `false`, submit and return immediately. |

### NeMo Customizer Trainer Adapter Configuration

The adapter handles communication with NeMo Customizer services.

```yaml
trainer_adapters:
  nemo_adapter:
    _type: nemo_customizer_trainer_adapter

    # Endpoint Configuration
    entity_host: https://nmp.example.com
    datastore_host: https://datastore.example.com
    hf_token: ""

    # Namespace and Dataset
    namespace: my-project
    dataset_name: nat-dpo
    dataset_output_dir: null
    create_namespace_if_missing: true

    # Customization Job
    customization_config: meta/llama-3.1-8b-instruct@v1.0.0+A100
    hyperparameters:
      training_type: dpo
      finetuning_type: all_weights
      epochs: 3
      batch_size: 4
      learning_rate: 5e-5
      dpo:
        ref_policy_kl_penalty: 0.1
        preference_loss_weight: 1.0
        preference_average_log_probs: false
        sft_loss_weight: 0.0

    # Prompt Formatting
    use_full_message_history: false

    # Deployment
    deploy_on_completion: false
    deployment_config:
      image_name: nvcr.io/nim/meta/llama-3.1-8b-instruct
      image_tag: latest
      gpu: 1
      deployment_name: null
      description: Fine-tuned model deployment

    # Polling
    poll_interval_seconds: 30.0
    deployment_timeout_seconds: 1800.0
```

#### Endpoint Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `entity_host` | `str` | **required** | Base URL for NeMo Entity Store (e.g., `https://nmp.example.com`). |
| `datastore_host` | `str` | **required** | Base URL for NeMo Datastore (e.g., `https://datastore.example.com`). |
| `hf_token` | `str` | `""` | HuggingFace token for datastore authentication. Can be empty if not required. |

#### Namespace and Dataset

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `namespace` | `str` | **required** | Namespace for organizing resources (datasets, models, deployments). |
| `dataset_name` | `str` | `"nat-dpo"` | Name for the training dataset. Must be unique within namespace. |
| `dataset_output_dir` | `str \| null` | `null` | Directory to save dataset JSONL files locally. If `null`, uses temporary directory. If specified, files are preserved for debugging. |
| `create_namespace_if_missing` | `bool` | `true` | If `true`, create namespace in entity store and datastore if it doesn't exist. |

#### Customization Job

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `customization_config` | `str` | **required** | Model configuration string (e.g., `meta/llama-3.1-8b-instruct@v1.0.0+A100`). Available `configs` can be listed via NeMo Customizer API. |

#### Hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training_type` | `"sft" \| "dpo"` | `"dpo"` | Training type. Use `"dpo"` for preference optimization. |
| `finetuning_type` | `"lora" \| "all_weights"` | `"all_weights"` | `"lora"` for parameter-efficient finetuning, `"all_weights"` for full model. |
| `epochs` | `int` | `3` | Number of training epochs over the dataset. |
| `batch_size` | `int` | `4` | Training batch size. |
| `learning_rate` | `float` | `5e-5` | Learning rate for optimizer. |

#### DPO-Specific Hyperparameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ref_policy_kl_penalty` | `float` | `0.1` | KL penalty coefficient (β in DPO objective). Controls how much the model can deviate from reference policy. Higher values = more conservative updates. |
| `preference_loss_weight` | `float` | `1.0` | Weight for the preference (DPO) loss term. |
| `preference_average_log_probs` | `bool` | `false` | If `true`, average log probabilities over sequence length. If `false`, sum log probabilities. |
| `sft_loss_weight` | `float` | `0.0` | Weight for optional SFT loss on chosen responses. Can help maintain response quality. |

#### Prompt Formatting

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_full_message_history` | `bool` | `false` | If `true`, include full conversation history as list of messages: `[{"role": "system", "content": "..."}, ...]`. If `false`, use only last message content as string. |

#### Deployment Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `deploy_on_completion` | `bool` | `false` | If `true`, automatically deploy the trained model after job completion. |
| `deployment_config.image_name` | `str` | `"nvcr.io/nim/meta/llama-3.1-8b-instruct"` | NIM container image name. |
| `deployment_config.image_tag` | `str` | `"latest"` | NIM container image tag. |
| `deployment_config.gpu` | `int` | `1` | Number of GPUs for deployment. |
| `deployment_config.deployment_name` | `str \| null` | `null` | Name for deployment. If `null`, auto-generated. |
| `deployment_config.description` | `str` | `"Fine-tuned model deployment"` | Description for the deployment. |

#### Polling Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `poll_interval_seconds` | `float` | `30.0` | Interval between job status checks. |
| `deployment_timeout_seconds` | `float` | `1800.0` | Maximum time to wait for deployment to be ready (30 minutes default). |

## Implementing TTC in Your Workflow

To generate DPO training data, your workflow must emit TTC (Test-Time Compute) intermediate steps with `TTCEventData`. Here's how to implement this:

### TTCEventData Structure

```python
from nat.data_models.intermediate_step import (
    IntermediateStepPayload,
    IntermediateStepType,
    TTCEventData,
)

# Create TTCEventData for each candidate
ttc_data = TTCEventData(
    turn_id="turn_0",           # Groups candidates competing for same prompt
    turn_index=0,               # Index of this turn in the episode
    candidate_index=idx,        # Index of this candidate within the turn
    input=messages,             # Prompt (string or list of OpenAI messages)
    output=response,            # Model's response
    score=candidate_score,      # Score for this candidate (higher = better)
)
```

### Emitting TTC Steps

```python
from nat.builder.context import Context

# Get the step manager from context
context = Context.get()
step_manager = context.intermediate_step_manager

# Emit TTC_END step for each candidate
step_manager.push_intermediate_step(
    IntermediateStepPayload(
        event_type=IntermediateStepType.TTC_END,
        name="dpo_candidate_move",  # Must match ttc_step_name in config
        data=ttc_data,
        metadata={"is_selected": is_best_candidate},
    )
)
```

### Complete Example: TTC Move Selector

```python
from nat.builder.context import Context
from nat.data_models.intermediate_step import (
    IntermediateStepPayload,
    IntermediateStepType,
    TTCEventData,
)

async def ttc_move_selector(
    prompt: str,
    candidates: list[str],
    scores: list[float],
    turn_id: str,
    turn_index: int,
) -> str:
    """
    Select best candidate and emit TTC steps for DPO training.

    Args:
        prompt: The input prompt
        candidates: List of candidate responses
        scores: Scores for each candidate (higher = better)
        turn_id: Unique identifier for this decision point
        turn_index: Index of this turn in the episode

    Returns:
        The best candidate response
    """
    context = Context.get()
    step_manager = context.intermediate_step_manager

    # Find best candidate
    best_idx = scores.index(max(scores))

    # Emit TTC_END step for each candidate
    for idx, (candidate, score) in enumerate(zip(candidates, scores)):
        ttc_data = TTCEventData(
            turn_id=turn_id,
            turn_index=turn_index,
            candidate_index=idx,
            input=prompt,
            output=candidate,
            score=score,
        )

        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                event_type=IntermediateStepType.TTC_END,
                name="dpo_candidate_move",
                data=ttc_data,
                metadata={"is_selected": idx == best_idx},
            )
        )

    return candidates[best_idx]
```

## How It Works

### Phase 1: Data Collection

The DPO trajectory builder collects preference data through the NeMo Agent toolkit evaluation system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DPO Trajectory Builder Flow                              │
│                                                                             │
│  start_run(run_id)                                                          │
│      │                                                                      │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Launch evaluation run                                                │  │
│  │                                                                       │  │
│  │  For each dataset example:                                            │  │
│  │    1. Execute workflow                                                │  │
│  │    2. Workflow emits TTC_END steps with TTCEventData                  │  │
│  │    3. Compute reward using configured evaluator                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  finalize(run_id)                                                           │
│      │                                                                      │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Process collected intermediate steps:                                │  │
│  │                                                                       │  │
│  │  1. Filter for TTC_END steps with configured name                     │  │
│  │  2. Extract TTCEventData (turn_id, candidate_index, score, etc.)      │  │
│  │  3. Group candidates by (example_id, turn_id)                         │  │
│  │  4. Generate preference pairs based on score differences              │  │
│  │  5. Build Trajectory objects with DPOItem episodes                    │  │
│  │  6. Group trajectories by example_id                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  Return TrajectoryCollection                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 2: Training Submission

The trainer adapter converts trajectories and submits to NeMo Customizer:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NeMo Customizer Trainer Adapter Flow                     │
│                                                                             │
│  submit(trajectories)                                                       │
│      │                                                                      │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Convert to JSONL format:                                             │  │
│  │                                                                       │  │
│  │  {                                                                    │  │
│  │    "prompt": "What move should I make?",                              │  │
│  │    "chosen_response": "I'll play X in the center...",                 │  │
│  │    "rejected_response": "I'll play X in the corner..."                │  │
│  │  }                                                                    │  │
│  │                                                                       │  │
│  │  Split: 80% training, 20% validation                                  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Upload to NeMo Datastore:                                            │  │
│  │                                                                       │  │
│  │  1. Create dataset repo via HuggingFace Hub API                       │  │
│  │  2. Register dataset in Entity Store                                  │  │
│  │  3. Upload training_file.jsonl and validation_file.jsonl              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Submit customization job:                                            │  │
│  │                                                                       │  │
│  │  client.customization.jobs.create(                                    │  │
│  │    config=customization_config,                                       │  │
│  │    dataset={name, namespace},                                         │  │
│  │    hyperparameters={training_type: dpo, ...}                          │  │
│  │  )                                                                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  Return TrainingJobRef                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 3: Monitoring and Completion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Training Monitoring Flow                                 │
│                                                                             │
│  wait_until_complete(job_ref)                                               │
│      │                                                                      │
│      ▼                                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Poll job status:                                                     │  │
│  │                                                                       │  │
│  │  while not done:                                                      │  │
│  │    status = client.customization.jobs.status(job_id)                  │  │
│  │    log status changes and progress                                    │  │
│  │    if status in [completed, failed, cancelled]: break                 │  │
│  │    sleep(poll_interval_seconds)                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼ (if deploy_on_completion and status == completed)                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Deploy trained model:                                                │  │
│  │                                                                       │  │
│  │  1. Create deployment config                                          │  │
│  │  2. Create model deployment                                           │  │
│  │  3. Wait for deployment to be ready                                   │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│      │                                                                      │
│      ▼                                                                      │
│  Return TrainingJobStatus                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Running DPO Training

### Basic Training

```bash
# Run DPO training with your configuration
nat finetune --config_file=configs/dpo_finetune.yml
```

### With Configuration Overrides

```bash
# Override number of data collection runs
nat finetune --config_file=configs/dpo_finetune.yml \
    -o trainers.nemo_trainer.num_runs 5

# Override training epochs
nat finetune --config_file=configs/dpo_finetune.yml \
    -o trainer_adapters.nemo_adapter.hyperparameters.epochs 5

# Override learning rate
nat finetune --config_file=configs/dpo_finetune.yml \
    -o trainer_adapters.nemo_adapter.hyperparameters.learning_rate 1e-5
```

### Monitoring Progress

During training, check:

1. **Console Output**: Shows data collection progress, pair counts, job status

```
INFO - Starting NeMo Customizer DPO workflow with 3 data collection runs
INFO - Starting data collection run 1/3
INFO - Run 1: Collected 50 trajectories, 120 DPO pairs, avg reward: 0.4523
INFO - Starting data collection run 2/3
INFO - Run 2: Collected 50 trajectories, 115 DPO pairs, avg reward: 0.4812
INFO - Starting data collection run 3/3
INFO - Run 3: Collected 50 trajectories, 118 DPO pairs, avg reward: 0.4701
INFO - Data collection complete: 150 trajectory groups, ~353 total DPO pairs from 3 runs
INFO - Deduplication: 353 -> 312 trajectories
INFO - Submitted training job: job_abc123
INFO - Job nemo_dpo_a1b2c3d4: Status -> 'running'
INFO - Job nemo_dpo_a1b2c3d4: Progress 25.0%
INFO - Job nemo_dpo_a1b2c3d4: Progress 50.0%
INFO - Job nemo_dpo_a1b2c3d4: Progress 75.0%
INFO - Job nemo_dpo_a1b2c3d4: Status -> 'completed'
INFO - Training completed with status: completed
```

2. **Output Files** (in `finetuning.output_dir`):
   - `data_collection_progress.jsonl`: Per-run metrics
   - `collection_history.json`: Complete collection history
   - `final_metrics.json`: Final training metrics

3. **NeMo Customizer UI**: Monitor job progress via the NeMo platform

## Dataset Format

The trainer adapter converts DPO pairs to JSONL format:

### Standard Format `(use_full_message_history: false)`

```json
{"prompt": "What's the best move in this position?", "chosen_response": "I'll play X in the center because...", "rejected_response": "I'll play X in the corner because..."}
{"prompt": "How should I respond to this attack?", "chosen_response": "I should defend by...", "rejected_response": "I should attack by..."}
```

### Full Message History Format `(use_full_message_history: true)`

```json
{"prompt": [{"role": "system", "content": "You are a chess expert."}, {"role": "user", "content": "What's the best move?"}], "chosen_response": "I recommend Nf3 because...", "rejected_response": "I recommend a4 because..."}
```

## Advanced Configuration

### Tuning DPO Hyperparameters

**KL Penalty (`ref_policy_kl_penalty`)**

The KL penalty (β) controls how much the model can deviate from the reference policy:

```yaml
hyperparameters:
  dpo:
    ref_policy_kl_penalty: 0.1  # Default: balanced exploration
    # ref_policy_kl_penalty: 0.01  # Lower: more aggressive updates
    # ref_policy_kl_penalty: 0.5   # Higher: more conservative updates
```

- **Lower values (0.01-0.05)**: Allow larger policy updates, faster learning but risk of instability
- **Higher values (0.2-0.5)**: More conservative updates, slower but more stable training

**SFT Loss Weight**

Adding SFT loss on chosen responses can help maintain response quality:

```yaml
hyperparameters:
  dpo:
    sft_loss_weight: 0.1  # Add 10% SFT loss
```

### Optimizing Data Collection

**Multiple Runs for Diversity**

Running multiple data collection passes generates diverse preference pairs:

```yaml
trainers:
  nemo_trainer:
    num_runs: 5  # More runs = more diverse data
```

**Filtering Weak Preferences**

Filter out pairs with small score differences:

```yaml
trajectory_builders:
  dpo_builder:
    min_score_diff: 0.1  # Only keep pairs with >0.1 score difference
```

**Limiting Pairs Per Turn**

For turns with many candidates, limit pairs to strongest preferences:

```yaml
trajectory_builders:
  dpo_builder:
    exhaustive_pairs: true
    max_pairs_per_turn: 5  # Keep top 5 pairs by score difference
```

### Automatic Model Deployment

Enable automatic deployment of trained models:

```yaml
trainer_adapters:
  nemo_adapter:
    deploy_on_completion: true
    deployment_config:
      image_name: nvcr.io/nim/meta/llama-3.1-8b-instruct
      image_tag: latest
      gpu: 2
      deployment_name: my-dpo-model
      description: DPO-finetuned agent model
    deployment_timeout_seconds: 3600  # 1 hour timeout
```

## Troubleshooting

### Connection Issues

**"Failed to connect to NeMo Customizer"**

1. Verify endpoints are correct:
   ```bash
   curl https://nmp.example.com/health
   curl https://datastore.example.com/health
   ```

2. Check authentication (HuggingFace token if required)

3. Verify network connectivity and firewall rules

### No Preference Pairs Generated

**"No trajectories collected from any run"**

1. **Check TTC step name**: Ensure `ttc_step_name` matches your workflow:
   ```yaml
   trajectory_builders:
     dpo_builder:
       ttc_step_name: dpo_candidate_move  # Must match workflow
   ```

2. **Verify TTCEventData is emitted**: Add logging to confirm steps are being pushed

3. **Check candidate scores**: If all candidates have same score, no pairs can be created

4. **Review `min_score_diff`**: Lower threshold if filtering too aggressively:
   ```yaml
   trajectory_builders:
     dpo_builder:
       min_score_diff: 0.0  # Accept all score differences
   ```

### Training Job Failures

**"Customization job failed"**

1. Check NeMo Customizer logs for detailed error messages

2. Verify dataset format is correct:
   ```bash
   # Check generated JSONL files
   cat .tmp/nat/finetuning/output/*/training_file.jsonl | head -5
   ```

3. Ensure model configuration is valid and available

4. Check GPU resources are available

### Deployment Issues

**"Deployment did not become ready within timeout"**

1. Increase timeout:
   ```yaml
   trainer_adapters:
     nemo_adapter:
       deployment_timeout_seconds: 3600  # 1 hour
   ```

2. Check NeMo deployment logs for errors

3. Verify GPU resources are available for deployment

4. Check deployment configuration matches model requirements

### Memory Issues

**"CUDA out of memory" during training**

1. Reduce batch size:
   ```yaml
   hyperparameters:
     batch_size: 2  # Reduce from default 4
   ```

2. Use LoRA instead of full-weight:
   ```yaml
   hyperparameters:
     finetuning_type: lora
   ```

3. Contact NeMo Customizer admin to allocate more GPU resources

## Examples

The `examples/finetuning/dpo_tic_tac_toe` directory contains a complete working example demonstrating:

- Tic-tac-toe game workflow with TTC move selection
- Custom scoring function for move quality
- Full DPO training configuration
- Training and evaluation datasets

See the example's README for detailed instructions.

## Best Practices

### Data Quality

1. **Meaningful Score Differences**: Ensure your scoring function produces meaningful distinctions between candidates
2. **Diverse Training Data**: Use multiple data collection runs and diverse input examples
3. **Balance Difficulty**: Include examples of varying difficulty levels

### Hyperparameter Selection

1. **Start Conservative**: Begin with default KL penalty (0.1) and adjust based on results
2. **Monitor Validation**: Track validation metrics to detect overfitting
3. **Iterate**: DPO often benefits from multiple rounds of training with fresh data

### Production Deployment

1. **Test Before Deploy**: Evaluate model quality before enabling automatic deployment
2. **Version Models**: Use descriptive deployment names for tracking
3. **Monitor Performance**: Track model performance in production and retrain as needed

<!-- path-check-skip-end -->

## See Also

- [Finetuning Concepts](concepts.md) - Core concepts and RL fundamentals
- [Extending the Finetuning Harness](../../extend/custom-components/finetuning.md) - Creating custom components
- [OpenPipe ART Integration](rl_with_openpipe.md) - Alternative RL training with ART
- [Custom Evaluators](../../extend/custom-components/custom-evaluator.md) - Creating reward functions
- [NeMo Customizer Documentation](https://docs.nvidia.com/nemo/microservices/latest/fine-tune/index.html) - Official NeMo Customizer documentation
