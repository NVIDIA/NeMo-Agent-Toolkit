<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ToolTalk Benchmark Evaluation

**Complexity:** 🟡 Intermediate

Evaluate NAT agent workflows against the [ToolTalk](https://github.com/microsoft/ToolTalk) multi-turn function calling benchmark. ToolTalk tests whether an agent can correctly select and execute API tools across multi-turn conversations with simulated database backends.

## Key Features

- **Multi-turn conversations**: Each scenario replays a full user-agent conversation with ground-truth tool calls
- **Simulated tool backends**: ToolTalk's `ToolExecutor` provides realistic database-backed tool responses
- **Native FC via `bind_tools()`**: Uses NAT's LangChain LLM with native function calling — no raw API calls
- **Built-in metrics**: Reports `recall`, `action_precision`, `bad_action_rate`, `success`, and `soft_success`
- **Easy + Full splits**: 29 easy conversations, 50 full conversations

## Table of Contents

- [Installation](#installation)
- [Set Up Environment](#set-up-environment)
- [Run Evaluation](#run-evaluation)
- [Understanding Results](#understanding-results)
- [Configuration Options](#configuration-options)

---

## Installation

From the root of the NeMo Agent Toolkit repository:

```bash
uv pip install -e examples/benchmarks/tooltalk
```

This installs the `nvidia-nat-benchmarks` package which provides the ToolTalk dataset loader, workflow, and evaluator.

### Prerequisites

The `tooltalk` package must be installed. It ships with the dataset and simulated databases:

```bash
pip install nvidia-tooltalk
```

---

## Set Up Environment

### 1. Set your NVIDIA API key

```bash
export NVIDIA_API_KEY=<your-nvidia-api-key>
```

Or add it to a `.env` file in your project root (NAT loads `.env` automatically).

### 2. Locate ToolTalk data paths

```bash
# Find the installed tooltalk data directories
python -c "
import tooltalk, os
base = os.path.dirname(tooltalk.__file__)
print(f'export TOOLTALK_DATABASE_DIR={os.path.join(base, \"data\", \"databases\")}')
print(f'export TOOLTALK_DATASET_DIR={os.path.join(base, \"data\", \"easy\")}')
"
```

**Expected output:**
```
export TOOLTALK_DATABASE_DIR=/path/to/.venv/lib/python3.11/site-packages/tooltalk/data/databases
export TOOLTALK_DATASET_DIR=/path/to/.venv/lib/python3.11/site-packages/tooltalk/data/easy
```

Set both variables:

```bash
export TOOLTALK_DATABASE_DIR=<path from above>
export TOOLTALK_DATASET_DIR=<path from above>
```

---

## Run Evaluation

### Easy split (29 conversations)

```bash
nat eval --config_file examples/benchmarks/tooltalk/configs/eval_easy.yml
```

**Expected output (during run):**
```
INFO - Starting evaluation run with config file: examples/benchmarks/tooltalk/configs/eval_easy.yml
INFO - Loaded 29 ToolTalk conversations from /path/to/tooltalk/data/easy
INFO - Shared workflow built (entry_function=None)
Running workflow:   0%|          | 0/29 [00:00<?, ?it/s]
Running workflow:   3%|▎         | 1/29 [00:45<21:00, 45.00s/it]
...
Running workflow: 100%|██████████| 29/29 [15:30<00:00, 32.07s/it]
INFO - ToolTalk evaluation complete: average_success=0.XXX across 29 conversations

=== EVALUATION SUMMARY ===
Workflow Status: COMPLETED (workflow_output.json)
Total Runtime: 930.00s

Per evaluator results:
| Evaluator   |   Avg Score | Output File          |
|-------------|-------------|----------------------|
| tooltalk    |       0.XXX | tooltalk_output.json |
```

### Full split (50 conversations)

Change `TOOLTALK_DATASET_DIR` to the `tooltalk` directory:

```bash
export TOOLTALK_DATASET_DIR=/path/to/.venv/lib/python3.11/site-packages/tooltalk/data/tooltalk
nat eval --config_file examples/benchmarks/tooltalk/configs/eval_easy.yml
```

---

## Understanding Results

Results are saved to `.tmp/nat/benchmarks/tooltalk/easy/`:

```bash
ls .tmp/nat/benchmarks/tooltalk/easy/
# config_original.yml  config_effective.yml  workflow_output.json  tooltalk_output.json
```

### Inspect per-conversation metrics

```bash
python -c "
import json
with open('.tmp/nat/benchmarks/tooltalk/easy/tooltalk_output.json') as f:
    data = json.load(f)
print(f'Average success rate: {data[\"average_score\"]:.3f}')
print(f'Total conversations: {len(data[\"eval_output_items\"])}')
for item in data['eval_output_items'][:3]:
    r = item['reasoning']
    print(f'  {item[\"id\"]}: recall={r[\"recall\"]:.2f} bad_actions={r[\"bad_actions\"]} success={r[\"success\"]}')
"
```

**Example output:**
```
Average success rate: 0.345
Total conversations: 29
  90a1a217-...: recall=1.00 bad_actions=2 success=0.0
  fc16280a-...: recall=1.00 bad_actions=0 success=1.0
  a3b4c5d6-...: recall=0.50 bad_actions=1 success=0.0
```

### Metrics explained

| Metric | Description |
|--------|-------------|
| `recall` | Fraction of ground-truth API calls that were correctly predicted |
| `action_precision` | Fraction of predicted actions that match ground truth |
| `bad_action_rate` | Fraction of actions that are successful but don't match ground truth |
| `success` | 1.0 only if recall=1.0 AND bad_action_rate=0.0 |
| `soft_success` | recall * (1 - bad_action_rate) — partial credit metric |

---

## Configuration Options

### Workflow settings

| Field | Default | Description |
|-------|---------|-------------|
| `api_mode` | `all` | Which tool docs to include: `exact`, `suite`, or `all` |
| `max_tool_calls_per_turn` | `10` | Safety limit per conversation turn |
| `disable_documentation` | `false` | Send empty tool descriptions (tests tool selection without docs) |

### Dataset settings

| Field | Default | Description |
|-------|---------|-------------|
| `file_path` | required | Path to ToolTalk data directory (easy or tooltalk split) |
| `database_dir` | required | Path to ToolTalk database directory |

### Tips for better results

- Use `api_mode: exact` to provide only the tools used in each conversation (easier for the LLM)
- Set `max_tool_calls_per_turn: 3` to reduce duplicate tool calls
- Use `temperature: 0.0` for deterministic tool selection
