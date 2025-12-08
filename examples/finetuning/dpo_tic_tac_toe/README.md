# DPO Tic-Tac-Toe: Preference Data Generation with Test Time Compute

This example demonstrates how to use NAT's Test Time Compute (TTC) harness to generate preference data for Direct Preference Optimization (DPO) finetuning.

## Overview

The workflow generates multiple candidate moves per turn for **both players** using TTC pipelines, scores each move using game-theoretic evaluation, and records all candidates as intermediate steps. This enables DPO data collection from ALL game turns, not just the trained player's turns.

This data can then be used to construct DPO preference pairs where:

- **Chosen response**: The move that was selected (highest score)
- **Rejected response**: Other candidate moves with lower scores

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DPO Tic-Tac-Toe Workflow                            │
│                                                                             │
│  workflow (dpo_tic_tac_toe)                                                 │
│    │                                                                        │
│    └── For EACH turn (trained player AND opponent), calls:                  │
│                                                                             │
│        ttc_move_selector (NAT Function) ─── trained or opponent version     │
│          │                                                                  │
│          ├── 1. SEARCH: move_searcher (TTC Strategy)                        │
│          │       │                                                          │
│          │       └── Calls choose_move (NAT Function) N times               │
│          │           - With LLM: generates LLM-based candidates             │
│          │           - Without LLM: generates random candidates             │
│          │                                                                  │
│          ├── 2. SCORE: move_scorer (TTC Strategy)                           │
│          │       │                                                          │
│          │       └── Evaluates each candidate using                         │
│          │           evaluate_board_for_player()                            │
│          │                                                                  │
│          ├── 3. SELECT: move_selector (TTC Strategy)                        │
│          │       │                                                          │
│          │       └── Chooses the highest-scoring move                       │
│          │                                                                  │
│          └── 4. RECORD: Writes intermediate steps for ALL candidates        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. choose_move NAT Function (`choose_move_function.py`)

A NAT Function that generates a single move for a given board state. Supports two modes:
- **LLM mode**: When `llm` is configured, uses the LLM to generate moves
- **Random mode**: When `llm` is omitted, generates random legal moves

```yaml
functions:
  # LLM-based move generation for trained player
  trained_choose_move:
    _type: choose_move
    llm: training_llm
    max_retries: 2

  # Random move generation for opponent
  random_choose_move:
    _type: choose_move
    # llm is null - generates random moves
```

### 2. multi_candidate_move_search TTC Strategy (`move_search_strategy.py`)

A TTC **SEARCH** strategy that generates N candidate moves by calling `choose_move` multiple times. Separate instances are configured for trained player and opponent.

```yaml
ttc_strategies:
  # For trained player (uses LLM-based choose_move)
  trained_move_searcher:
    _type: multi_candidate_move_search
    choose_move_fn: trained_choose_move
    num_candidates: 3

  # For opponent (uses random choose_move)
  random_move_searcher:
    _type: multi_candidate_move_search
    choose_move_fn: random_choose_move
    num_candidates: 3
```

### 3. board_position_scorer TTC Strategy (`board_position_scorer.py`)

A TTC **SCORING** strategy that evaluates moves using game-theoretic position analysis (alpha-beta minimax with heuristic evaluation).

```yaml
ttc_strategies:
  move_scorer:
    _type: board_position_scorer
```

### 4. ttc_move_selector NAT Function (`ttc_move_selector_function.py`)

A NAT Function that wraps the complete TTC pipeline (search → score → select) and records all candidates as intermediate steps. Separate instances are configured for trained player and opponent.

```yaml
functions:
  # TTC pipeline for trained player
  trained_ttc_move_selector:
    _type: ttc_move_selector
    search: trained_move_searcher
    scorer: move_scorer
    selector: move_selector

  # TTC pipeline for opponent
  random_ttc_move_selector:
    _type: ttc_move_selector
    search: random_move_searcher
    scorer: move_scorer
    selector: move_selector
```

### 5. Main Workflow (`dpo_workflow.py`)

The main workflow that plays games, using TTC pipelines for BOTH players. This enables DPO data collection from all game turns.

```yaml
workflow:
  _type: dpo_tic_tac_toe
  trained_ttc_move_selector_fn: trained_ttc_move_selector
  opponent_ttc_move_selector_fn: random_ttc_move_selector
```

## Intermediate Step Data Structure

Each candidate move is recorded with full context for DPO training:

```python
{
    "turn_id": "turn_0_abc12345",           # Unique per turn
    "move_id": "turn_0_abc12345_move_0",    # Unique per move
    "board_state_before": [[0,0,0],...],    # Board before move
    "prompt": "   |   |   \n-----------\n   |   |   \n-----------\n   |   |   ",
    "move": {"row": 1, "col": 1},           # The move
    "score": 0.85,                          # evaluate_board_for_player value
    "is_selected": True,                    # Whether this was chosen
    "raw_llm_response": "<move>...",        # Original LLM output (completion)
    "player_symbol": "X",
    "player_value": 1
}
```

The `prompt` field contains the board state string (output of `board_to_str`), representing the last user input to the model.

## Prerequisites

- Python 3.11+
- A running LLM inference server (vLLM, OpenAI-compatible endpoint)

## Installation

```bash
cd examples/finetuning/dpo_tic_tac_toe
uv sync
```

## Running the Example

### 1. Start an LLM Server

Using vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --port 8000
```

### 2. Run Evaluation

```bash
nat eval --config_file=configs/config.yml
```

This will:
1. Play games using the TTC pipeline to generate multiple candidates per turn
2. Score and select the best moves
3. Record all candidates as intermediate steps
4. Output evaluation results with DPO preference pairs

## Output Format

The `dpo_collector` evaluator extracts preference pairs from intermediate steps:

```json
{
  "dpo_pairs": [
    {
      "turn_id": "turn_0_abc12345",
      "prompt": "   |   |   \n-----------\n   |   |   \n-----------\n   |   |   ",
      "chosen": {
        "move": {"row": 1, "col": 1},
        "response": "<move><row>2</row><col>2</col></move>",
        "score": 0.85
      },
      "rejected": {
        "move": {"row": 0, "col": 0},
        "response": "<move><row>1</row><col>1</col></move>",
        "score": 0.72
      },
      "score_diff": 0.13
    }
  ]
}
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_candidates` | Number of candidate moves per turn (in move_searcher) | 3 |
| `max_retries` | Max retries for XML parsing (in choose_move with LLM) | 2 |
| `llm` | LLM for move generation in choose_move (null=random) | null |

## DPO Trajectory Builder

The DPO Trajectory Builder (`dpo_traj_builder`) collects preference pairs from the intermediate steps and converts them to NAT's trajectory format for DPO finetuning.

### Configuration

```yaml
trajectory_builders:
  dpo_builder:
    _type: dpo_traj_builder
    # Name of the CUSTOM intermediate step to collect
    custom_step_name: dpo_candidate_move
    # Generate all pairwise comparisons (not just best vs worst)
    exhaustive_pairs: true
    # Minimum score difference to create a pair
    min_score_diff: 0.01
    # Maximum pairs per turn (null = unlimited)
    max_pairs_per_turn: 5
```

### Trajectory Builder Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `custom_step_name` | Name of CUSTOM intermediate step to collect | `dpo_candidate_move` |
| `exhaustive_pairs` | Generate all pairwise comparisons vs just best/worst | `true` |
| `min_score_diff` | Minimum score difference for a valid pair | `0.0` |
| `max_pairs_per_turn` | Max pairs per turn (null = unlimited) | `null` |
| `turn_id_key` | Metadata key for turn identifier | `turn_id` |
| `score_key` | Metadata key for candidate score | `score` |
| `prompt_key` | Metadata key for input prompt | `prompt` |
| `response_key` | Metadata key for model response | `raw_llm_response` |
| `reward_from_score_diff` | Use score diff as reward (vs chosen score) | `true` |

### Pair Generation Modes

**Exhaustive Mode** (`exhaustive_pairs: true`):
For candidates [A, B, C] with scores [0.9, 0.7, 0.5]:
- Generates pairs: (A>B), (A>C), (B>C)
- Maximum N*(N-1)/2 pairs per turn

**Best vs Worst Mode** (`exhaustive_pairs: false`):
For candidates [A, B, C] with scores [0.9, 0.7, 0.5]:
- Generates only: (A>C)
- One pair per turn

### Output Format

Each trajectory contains:
- **episode**: `[user_prompt, assistant_chosen_response]`
- **reward**: Score difference (chosen - rejected)
- **metadata**: Contains `rejected_response`, scores, turn info

```python
Trajectory(
    episode=[
        EpisodeItem(role=USER, content="<board_state>"),
        EpisodeItem(role=ASSISTANT, content="<chosen_move>"),
    ],
    reward=0.15,  # score_diff
    metadata={
        "dpo_type": "preference_pair",
        "rejected_response": "<rejected_move>",
        "chosen_score": 0.85,
        "rejected_score": 0.70,
        "score_diff": 0.15,
        "example_id": "...",
        "turn_id": "...",
    }
)
```

## Phase 3: DPO Trainer Adapter

The next phase will implement a trainer adapter to submit DPO trajectories to a training backend (e.g., NeMo Customizer) for model finetuning.

## See Also

- [Finetuning Concepts](../../../docs/source/reference/finetuning/concepts.md)
- [Test Time Compute](../../../docs/source/reference/test-time-compute.md)
- [RL with OpenPipe ART](../rl_with_openpipe_art/) - Original RL example
