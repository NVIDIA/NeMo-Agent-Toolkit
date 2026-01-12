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

# DPO Tic-Tac-Toe: Preference Learning with NeMo Customizer

This example demonstrates how to use the NeMo Agent toolkit Test Time Compute (TTC) pipeline to generate preference data for Direct Preference Optimization (DPO) training, and submit training jobs to NVIDIA NeMo Customizer.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Architecture](#architecture)
- [How Move Scoring Works](#how-move-scoring-works)
- [Installation](#installation)
- [Configuration Reference](#configuration-reference)
- [Running the Example](#running-the-example)
- [Understanding the Output](#understanding-the-output)
- [Troubleshooting](#troubleshooting)

## Overview

The workflow generates multiple candidate moves per turn for **both players** using TTC pipelines, scores each move using game-theoretic evaluation with alpha-beta pruning, and records all candidates as intermediate steps. This enables DPO data collection from ALL game turns.

The collected preference data is then submitted to NeMo Customizer for DPO training, and optionally deployed as a NIM endpoint.

### What is DPO?

Direct Preference Optimization (DPO) is a technique for aligning language models with human preferences without requiring a separate reward model. Instead of training a reward model and then using reinforcement learning, DPO directly optimizes the model using preference pairs:

- **Chosen response**: The move that was selected (highest score)
- **Rejected response**: Other candidate moves with lower scores

The model learns to prefer responses similar to the chosen examples while avoiding patterns in rejected examples.

## Prerequisites

### 1. Python Environment

- Python 3.11 or higher
- `uv` package manager (recommended)

### 2. NVIDIA NeMo Microservices Platform

This example requires access to the following NeMo Microservices:

#### NeMo Customizer Service
The customization service handles DPO/SFT training jobs.

- **Endpoint**: Your NeMo Customizer URL (e.g., `https://nmp.example.com`)
- **Purpose**: Submits and monitors training jobs
- **Required API**: Customization Jobs API (`/v1/customization/jobs`)

#### NeMo Entity Store
The entity store manages namespaces and metadata.

- **Endpoint**: Same as Customizer or dedicated URL
- **Purpose**: Namespace management, model registration
- **Required API**: Namespaces API (`/v1/namespaces`)

#### NeMo Datastore
The datastore handles dataset upload and storage.

- **Endpoint**: Your Datastore URL (e.g., `https://datastore.example.com`)
- **Purpose**: Upload training datasets, store model artifacts
- **Required API**: Datasets API, Upload API

#### NIM Deployment Service (Optional)
For automatic model deployment after training.

- **Endpoint**: Same as Customizer
- **Purpose**: Deploy trained models as NIM endpoints
- **Required API**: Model Deployments API (`/v1/deployment/model-deployments`)

### 3. Model Configuration

You need a valid customization configuration string for your target model. Available configurations can be listed via the NeMo Customizer API:

```bash
# List available customization configs
curl -X GET "https://your-nmp-host/v1/customization/configs" \
  -H "Authorization: Bearer $NGC_API_KEY"
```

Common configurations:
- `meta/llama-3.1-8b-instruct@v1.0.0+A100` - Llama 3.1 8B on A100 GPUs
- `meta/llama-3.2-1b-instruct@v1.0.0+A100` - Llama 3.2 1B on A100 GPUs

### 4. LLM Inference Endpoint

For move generation during data collection, you need an OpenAI-compatible LLM endpoint:

- **Local**: vLLM, text-generation-inference, Ollama
- **Cloud**: Any OpenAI-compatible API

### 5. Authentication

Set the following environment variables:

```bash
# NGC API key for NeMo services
export NGC_API_KEY="your-ngc-api-key"

# HuggingFace token (if required by datastore)
export HF_TOKEN="your-hf-token"

# OpenAI-compatible API key for inference
export OPENAI_API_KEY="unused-default-key"

# NeMo Customizer service endpoints
export CUSTOMIZER_HOST="https://your-nmp-host"
export DATASTORE_HOST="https://your-datastore-host"
export CUSTOMIZER_NIM_URL="https://your-nim-deployment-host"
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DPO Tic-Tac-Toe Pipeline                            │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    1. DATA COLLECTION PHASE                          │   │
│  │                                                                      │   │
│  │  workflow (dpo_tic_tac_toe)                                          │   │
│  │    │                                                                 │   │
│  │    └── For EACH turn (trained player AND opponent):                  │   │
│  │                                                                      │   │
│  │        ttc_move_selector (Function)                                  │   │
│  │          │                                                           │   │
│  │          ├── 1. SEARCH: move_searcher                                │   │
│  │          │       └── Calls choose_move N times                       │   │
│  │          │           (LLM-based or random)                           │   │
│  │          │                                                           │   │
│  │          ├── 2. SCORE: board_position_scorer                         │   │
│  │          │       └── Alpha-beta Minimax evaluation                   │   │
│  │          │                                                           │   │
│  │          ├── 3. SELECT: best_of_n_selection                          │   │
│  │          │       └── Choose highest-scoring move                     │   │
│  │          │                                                           │   │
│  │          └── 4. RECORD: Emit CUSTOM intermediate steps               │   │
│  │                  └── All candidates with scores                      │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    2. TRAJECTORY BUILDING PHASE                      │   │
│  │                                                                      │   │
│  │  dpo_traj_builder                                                    │   │
│  │    │                                                                 │   │
│  │    ├── Filter CUSTOM_END steps by name                               │   │
│  │    ├── Group candidates by turn_id                                   │   │
│  │    ├── Generate preference pairs based on scores                     │   │
│  │    └── Output: List of DPO trajectories                              │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    3. TRAINING PHASE                                 │   │
│  │                                                                      │   │
│  │  nemo_customizer_trainer_adapter                                     │   │
│  │    │                                                                 │   │
│  │    ├── Format trajectories as NeMo DPO dataset                       │   │
│  │    ├── Upload dataset to NeMo Datastore                              │   │
│  │    ├── Submit training job to NeMo Customizer                        │   │
│  │    ├── Poll until training completes                                 │   │
│  │    └── (Optional) Deploy trained model as NIM                        │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How Move Scoring Works

The scoring system uses **game-theoretic position evaluation** combining heuristic features with alpha-beta Minimax search. This provides accurate move scoring without requiring an LLM judge.

### Alpha-Beta Pruning Algorithm

Alpha-beta pruning is an optimization of the Minimax algorithm that eliminates branches that cannot possibly affect the final decision. It maintains two values:

- **Alpha (α)**: The best value that the maximizer (current player) can guarantee
- **Beta (β)**: The best value that the minimizer (opponent) can guarantee

When `α ≥ β`, the current branch is pruned because the opponent would never allow this position.

```python
def solve_outcome(board, side_to_move, alpha=-1.0, beta=1.0):
    """
    Game-theoretic outcome with alpha-beta pruning.

    Returns:
      +1  -> Current player can force a win
       0  -> Perfect play leads to draw
      -1  -> Current player will lose with best play
    """
    # Check terminal states
    winner = check_winner(board)
    if winner == player_val:
        return 1.0
    elif winner == -player_val:
        return -1.0
    elif is_draw(board):
        return 0.0

    if side_to_move == player_val:
        # Maximizing player
        best = -1.0
        for move in available_moves(board):
            apply_move(board, move, side_to_move)
            value = solve_outcome(board, -side_to_move, alpha, beta)
            undo_move(board, move)

            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # Beta cutoff - opponent won't allow this
        return best
    else:
        # Minimizing player (opponent)
        best = 1.0
        for move in available_moves(board):
            apply_move(board, move, side_to_move)
            value = solve_outcome(board, -side_to_move, alpha, beta)
            undo_move(board, move)

            best = min(best, value)
            beta = min(beta, best)
            if alpha >= beta:
                break  # Alpha cutoff - we already have better
        return best
```

### Score Ranges

The `evaluate_board_for_player` function returns scores in different ranges:

| Situation | Score Range | Meaning |
|-----------|-------------|---------|
| Forced loss | `0.0` | Player will lose with perfect opponent play |
| Uncertain | `[0, 1]` | No forced outcome; uses heuristic evaluation |
| Forced future win | `(10, 11]` | Player can force a win (base + 10) |
| Immediate win | `(15, 16]` | Player has already won (base + 15) |

### Heuristic Features

For non-terminal positions without forced outcomes, the scorer uses these features:

1. **Two-in-a-row threats**: Lines with 2 of our pieces and no opponent pieces (+4 weight)
2. **One-in-a-row potential**: Lines with 1 of our pieces and no opponent pieces (+1.5 weight)
3. **Center control**: Occupying the center square (+1.5 weight)
4. **Corner control**: Occupying corner squares (+0.75 weight each)
5. **Edge control**: Occupying edge squares (+0.25 weight each)

## Installation

This example is meant to be run using a NeMo Agent Toolkit installation from source. You 
can follow the [NeMo Agent Toolkit Installation Guide](../../../docs/source/get-started/installation.md) to set up your environment.

Then: 

```bash
uv pip install -e examples/finetuning/dpo_tic_tac_toe
```

## Configuration Reference

The configuration is defined in `configs/config.yml`. Here's a complete reference:

### LLM Configuration

```yaml
llms:
  training_llm:
    _type: openai
    model_name: meta-llama/Llama-3.1-8B-Instruct
    base_url: http://localhost:8000/v1
    # Or use a deployed NIM endpoint:
    # base_url: https://nim.example.com/v1
```

### Functions

```yaml
functions:
  # LLM-based move generation for trained player
  trained_choose_move:
    _type: choose_move
    llm: training_llm
    max_retries: 2

  # TTC pipeline for trained player
  trained_ttc_move_selector:
    _type: ttc_move_selector
    search: trained_move_searcher
    scorer: move_scorer
    selector: move_selector

  # Random move generation for opponent (no LLM)
  random_choose_move:
    _type: choose_move
    # llm is null - generates random legal moves

  # TTC pipeline for opponent
  random_ttc_move_selector:
    _type: ttc_move_selector
    search: random_move_searcher
    scorer: move_scorer
    selector: move_selector
```

### TTC Strategies

```yaml
ttc_strategies:
  # SEARCH strategy for trained player
  trained_move_searcher:
    _type: multi_candidate_move_search
    choose_move_fn: trained_choose_move
    num_candidates: 3  # Generate 3 candidates per turn

  # SEARCH strategy for opponent
  random_move_searcher:
    _type: multi_candidate_move_search
    choose_move_fn: random_choose_move
    num_candidates: 3

  # SCORING strategy (shared)
  move_scorer:
    _type: board_position_scorer

  # SELECTION strategy (shared)
  move_selector:
    _type: best_of_n_selection
```

### Workflow Configuration

```yaml
workflow:
  _type: dpo_tic_tac_toe
  trained_ttc_move_selector_fn: trained_ttc_move_selector
  opponent_ttc_move_selector_fn: random_ttc_move_selector
```

### Evaluation Configuration

```yaml
eval:
  general:
    max_concurrency: 8
    output_dir: .tmp/nat/dpo_tic_tac_toe/eval
    dataset:
      _type: json
      file_path: examples/finetuning/dpo_tic_tac_toe/data/data.json

  evaluators:
    game_outcome:
      _type: dpo_game_outcome
```

### DPO Trajectory Builder

```yaml
trajectory_builders:
  dpo_builder:
    _type: dpo_traj_builder
    # Name of CUSTOM intermediate step to collect
    custom_step_name: dpo_candidate_move
    # Generate all pairwise comparisons
    exhaustive_pairs: true
    # Minimum score difference for valid pair
    min_score_diff: 0.01
    # Maximum pairs per turn (null = unlimited)
    max_pairs_per_turn: 5
    # Use score difference as reward
    reward_from_score_diff: true
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `custom_step_name` | Name of CUSTOM step to filter | `dpo_candidate_move` |
| `exhaustive_pairs` | All pairs vs best/worst only | `true` |
| `min_score_diff` | Minimum score difference | `0.0` |
| `max_pairs_per_turn` | Max pairs per turn | `null` (unlimited) |
| `reward_from_score_diff` | Reward = `score_diff` vs `chosen_score` | `true` |
| `require_multiple_candidates` | Skip single-candidate turns | `true` |

### NeMo Customizer Trainer Adapter

```yaml
trainer_adapters:
  nemo_customizer_trainer_adapter:
    _type: nemo_customizer_trainer_adapter

    # === NeMo Service Endpoints ===
    entity_host: ${CUSTOMIZER_HOST}
    datastore_host: ${DATASTORE_HOST}

    # === Namespace and Dataset ===
    namespace: nat-dpo-test
    dataset_name: nat-dpo
    dataset_output_dir: .tmp/output/datasets
    create_namespace_if_missing: true

    # === Model Configuration ===
    customization_config: meta/llama-3.1-8b-instruct@v1.0.0+A100

    # === Training Hyperparameters ===
    hyperparameters:
      training_type: dpo
      finetuning_type: all_weights  # or "lora"
      epochs: 5
      batch_size: 8
      learning_rate: 0.00005
      dpo:
        ref_policy_kl_penalty: 0.1
        preference_loss_weight: 1.0
        preference_average_log_probs: false
        sft_loss_weight: 0.0

    # === Prompt Formatting ===
    use_full_message_history: false

    # === Deployment (Optional) ===
    deploy_on_completion: true
    deployment_config:
      image_name: nvcr.io/nim/meta/llama-3.1-8b-instruct
      image_tag: latest
      gpu: 2
      deployment_name: nat_dpo_tic_tac_toe_model
      description: Fine-tuned model by NAT

    # === Polling Configuration ===
    poll_interval_seconds: 30.0
    deployment_timeout_seconds: 1800.0
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `entity_host` | NeMo Entity Store URL | (required) |
| `datastore_host` | NeMo Datastore URL | (required) |
| `namespace` | Resource namespace | (required) |
| `customization_config` | Model config string | (required) |
| `dataset_name` | Training dataset name | `nat-dpo` |
| `dataset_output_dir` | Local dataset save path | `null` (temp) |
| `use_full_message_history` | Include full chat history | `false` |
| `deploy_on_completion` | Auto-deploy after training | `false` |
| `poll_interval_seconds` | Job status poll interval | `30.0` |
| `deployment_timeout_seconds` | Max deployment wait time | `1800.0` |

### NeMo Customizer Trainer

```yaml
trainers:
  nemo_customizer_trainer:
    _type: nemo_customizer_trainer
    num_runs: 1
    continue_on_collection_error: true
    deduplicate_pairs: true
    wait_for_completion: true
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_runs` | Data collection iterations | `1` |
| `continue_on_collection_error` | Continue if collection fails | `false` |
| `deduplicate_pairs` | Remove duplicate DPO pairs | `true` |
| `max_pairs` | Max pairs for training | `null` (all) |
| `wait_for_completion` | Wait for training to finish | `true` |

### Finetuning Configuration

```yaml
finetuning:
  enabled: true
  trainer: nemo_customizer_trainer
  trajectory_builder: dpo_builder
  trainer_adapter: nemo_customizer_trainer_adapter
  output_dir: ./.tmp/nat/finetuning/dpo_tic_tac_toe
```

## Running the Example

### Step 1: Start an LLM Server (for data collection)

Using vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

Or using a pre-deployed NIM endpoint - update `base_url` in config.

### Step 2: Run Evaluation Only (without training)

To test data collection without submitting training jobs:

```bash
# Run evaluation and collect DPO data
nat eval --config_file examples/finetuning/dpo_tic_tac_toe/configs/config.yml

# Results saved to .tmp/nat/dpo_tic_tac_toe/eval/
```

This will:
1. Play games using TTC pipeline
2. Generate and score multiple candidates per turn
3. Record all candidates as intermediate steps
4. Output evaluation metrics

### Step 3: Run Full Finetuning Pipeline

To collect data and submit training to NeMo Customizer:

```bash
# Set required environment variables
export NGC_API_KEY="your-ngc-api-key"

# Run finetuning pipeline
nat finetune --config_file examples/finetuning/dpo_tic_tac_toe/configs/config.yml
```

This will:
1. Run the trajectory builder to collect DPO data
2. Format data as NeMo-compatible JSONL
3. Upload dataset to NeMo Datastore
4. Submit DPO training job to NeMo Customizer
5. Poll until training completes
6. (Optional) Deploy trained model as NIM endpoint

### Step 4: Monitor Training Progress

Check training job status:

```bash
# List jobs in namespace
curl -X GET "https://your-nmp-host/v1/customization/jobs?namespace=nat-dpo-test" \
  -H "Authorization: Bearer $NGC_API_KEY"

# Get specific job status
curl -X GET "https://your-nmp-host/v1/customization/jobs/{job_id}" \
  -H "Authorization: Bearer $NGC_API_KEY"
```

## Understanding the Output

### Intermediate Step Structure

Each candidate move is recorded with:

```python
{
    "turn_id": "turn_0_abc12345",           # Unique per turn
    "turn_index": 0,                         # Turn number in game
    "candidate_index": 0,                    # Candidate number (0, 1, 2...)
    "board_state_before": [[0,0,0],...],    # Board before move
    "prompt": "  1 2 3\n1 _ _ _\n...",      # Board as string
    "move": {"row": 1, "col": 1},           # The move
    "score": 10.85,                          # Position evaluation
    "is_selected": true,                     # Whether chosen
    "raw_llm_response": "<move>...",        # LLM output
    "player_symbol": "X",
    "player_value": 1
}
```

### DPO Dataset Format

The training dataset is formatted as JSONL:

```json
{
  "prompt": [
    {"role": "system", "content": "You are playing Tic-Tac-Toe..."},
    {"role": "user", "content": "  1 2 3\n1 _ _ _\n2 _ _ _\n3 _ _ _"}
  ],
  "chosen_response": "<move><row>2</row><col>2</col></move>",
  "rejected_response": "<move><row>1</row><col>1</col></move>"
}
```

### Evaluation Metrics

The `dpo_game_outcome` evaluator reports:

- **Win rate**: Percentage of games won by trained player
- **Loss rate**: Percentage of games lost
- **Draw rate**: Percentage of games ending in draw
- **Average game length**: Mean number of turns per game

## Evaluating Your Trained Model

First, collect the name of the deployed model from the output of the finetuning step.

The ID of the deployed model will look something like: `default/meta-llama-3.1-8b-instruct-nat-dpo-all_weights@cust-XYZ`.
Export the name of the model, which is every thing before the `@` symbol:

```bash
export CUSTOMIZER_LLM_MODEL_NAME="default/meta-llama-3.1-8b-instruct-nat-dpo-all_weights"
```

Then, in the same terminal, run evaluation:

```bash
nat eval --config_file examples/finetuning/dpo_tic_tac_toe/configs/config_after_training.yml
```

## Troubleshooting

### Common Issues

#### 1. "Namespace not found" Error

**Cause**: The namespace doesn't exist in NeMo services.

**Solution**: Either create the namespace manually or set `create_namespace_if_missing: true` in config.

```yaml
trainer_adapters:
  nemo_customizer_trainer_adapter:
    create_namespace_if_missing: true
```

#### 2. "No preference pairs generated" Warning

**Cause**: No valid DPO pairs met the filtering criteria.

**Solutions**:
- Lower `min_score_diff` threshold
- Increase `num_candidates` in move searcher
- Check that CUSTOM intermediate steps are being emitted

#### 3. Training Job Fails

**Cause**: Various - check job logs.

**Debug steps**:
```bash
# Get job details with error message
curl -X GET "https://your-nmp-host/v1/customization/jobs/{job_id}" \
  -H "Authorization: Bearer $NGC_API_KEY" | jq '.status_details'
```

Common causes:
- Invalid `customization_config` string
- Insufficient GPU resources
- Dataset format issues

#### 4. Deployment Timeout

**Cause**: Model deployment taking longer than `deployment_timeout_seconds`.

**Solution**: Increase timeout or check deployment service health:

```yaml
trainer_adapters:
  nemo_customizer_trainer_adapter:
    deployment_timeout_seconds: 3600.0  # 1 hour
```

#### 5. TTCEventData Fields Missing

**Cause**: Serialization issue with intermediate steps.

**Solution**: Ensure you're using the latest NeMo Agent toolkit version with `SerializeAsAny` fix in `IntermediateStepPayload`.

### Debug Logging

Enable verbose logging:

```bash
export NAT_LOG_LEVEL=DEBUG
nat finetune --config_file=configs/config.yml
```

Or in Python:

```python
import logging
logging.getLogger("nat").setLevel(logging.DEBUG)
logging.getLogger("nat.plugins.customizer").setLevel(logging.DEBUG)
```

<!-- path-check-skip-end -->

## See Also

- [Finetuning Concepts](../../../docs/source/improve-workflows/finetuning/concepts.md) - NeMo Agent toolkit finetuning architecture
- [Test Time Compute](../../../docs/source/improve-workflows/test-time-compute.md) - TTC pipeline reference
- [RL with OpenPipe ART](../rl_with_openpipe_art/) - Alternative RL-based finetuning example
