<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Galileo Agent Leaderboard v2 Evaluation

**Complexity:** 🟡 Intermediate

Evaluate NAT agent workflows against the [Galileo Agent Leaderboard v2](https://huggingface.co/datasets/galileo-ai/agent-leaderboard-v2) benchmark. This benchmark tests whether an agent can select the correct tools for real-world use cases across multiple domains.

## Key Features

- **5 domains**: Banking, Healthcare, Insurance, Investment, Telecom
- **Tool stub execution**: All domain tools are registered as stubs — the agent selects tools without executing real backends
- **Tool Selection Quality (TSQ)**: F1 score between predicted and expected tool calls
- **HuggingFace integration**: Dataset downloads automatically from `galileo-ai/agent-leaderboard-v2`
- **Multi-domain evaluation**: Evaluate across one or all domains in a single run

## Table of Contents

- [Installation](#installation)
- [Set Up Environment](#set-up-environment)
- [Option A: Download Dataset First](#option-a-download-dataset-first)
- [Option B: Auto-Download from HuggingFace](#option-b-auto-download-from-huggingface)
- [Run Evaluation](#run-evaluation)
- [Understanding Results](#understanding-results)
- [All Domains Evaluation](#all-domains-evaluation)

---

## Installation

```bash
uv pip install -e examples/benchmarks/agent_leaderboard
```

This installs the `datasets` library for HuggingFace access.

---

## Set Up Environment

```bash
export NVIDIA_API_KEY=<your-nvidia-api-key>
```

---

## Option A: Download Dataset First

Use the download script to fetch and transform the dataset:

```bash
python examples/dynamo_integration/scripts/download_agent_leaderboard_v2.py \
  --output-dir data/agent_leaderboard \
  --domains banking
```

**Expected output:**
```
INFO - Loading agent leaderboard v2 dataset from Hugging Face...
INFO - Loading domain: banking
INFO - Loaded 20 tools, 20 personas, 100 scenarios for banking
INFO - Saved 100 entries to data/agent_leaderboard/agent_leaderboard_v2_banking.json
INFO - Saved raw data to data/agent_leaderboard/raw/banking
```

Then set the data path:

```bash
export AGENT_LEADERBOARD_DATA=data/agent_leaderboard/agent_leaderboard_v2_banking.json
```

---

## Option B: Auto-Download from HuggingFace

If no local file is found, the dataset loader downloads directly from HuggingFace. Just point `file_path` to a non-existent path and the `domains` config will be used to download:

```yaml
dataset:
  _type: agent_leaderboard
  file_path: ./data/auto_download.json  # Will trigger HF download
  domains: [banking]
```

---

## Run Evaluation

### Banking domain (quick test with 10 scenarios)

```bash
export AGENT_LEADERBOARD_LIMIT=10
nat eval --config_file examples/benchmarks/agent_leaderboard/configs/eval_banking.yml
```

**Expected output:**
```
INFO - Starting evaluation run with config file: .../eval_banking.yml
INFO - Loaded 10 entries from data/agent_leaderboard/agent_leaderboard_v2_banking.json
INFO - Shared workflow built (entry_function=None)
Running workflow: 100%|██████████| 10/10 [03:20<00:00, 20.00s/it]
INFO - TSQ evaluation complete: avg_f1=0.650 across 10 scenarios

=== EVALUATION SUMMARY ===
| Evaluator |   Avg Score | Output File     |
|-----------|-------------|-----------------|
| tsq       |       0.650 | tsq_output.json |
```

### Full banking evaluation

```bash
unset AGENT_LEADERBOARD_LIMIT
nat eval --config_file examples/benchmarks/agent_leaderboard/configs/eval_banking.yml
```

---

## Understanding Results

### The `agent_leaderboard_tsq` evaluator

This example uses the **Tool Selection Quality (TSQ)** evaluator (`_type: agent_leaderboard_tsq` in the eval config). It compares the tool calls the agent made (captured by the workflow via `ToolIntentBuffer`) against the expected tool calls derived from the scenario's user goals.

The evaluator computes an **F1 score** between predicted and expected tool sets:
- **Precision** = (correctly predicted tools) / (total predicted tools)
- **Recall** = (correctly predicted tools) / (total expected tools)
- **F1** = 2 × precision × recall / (precision + recall)

Tool names are normalized before comparison (case-insensitive, underscores/hyphens stripped, module prefixes removed) so that `banking_tools__get_account_balance` matches `get_account_balance`.

The evaluator is configured in the YAML under `eval.evaluators`:

```yaml
evaluators:
  tsq:
    _type: agent_leaderboard_tsq
    tool_weight: 1.0          # Weight for tool selection F1 (default: 1.0)
    parameter_weight: 0.0     # Weight for parameter accuracy (default: 0.0)
```

The final score per item is `tool_weight × tool_f1 + parameter_weight × param_accuracy`. With default weights, only tool selection matters.

### Per-item metrics

Each item in the evaluator output contains:

| Field | Description |
|-------|-------------|
| `tool_selection_f1` | F1 score between predicted and expected tool names |
| `parameter_accuracy` | Parameter correctness (placeholder — future enhancement) |
| `predicted_tools` | Normalized list of tools the agent called |
| `expected_tools` | Normalized list of tools expected from user goals |
| `num_predicted` | Total tool call intents captured |
| `num_expected` | Total expected tool calls from ground truth |

### Inspect results

```bash
python -c "
import json
with open('.tmp/nat/benchmarks/agent_leaderboard/banking/tsq_output.json') as f:
    data = json.load(f)
print(f'Average TSQ F1: {data[\"average_score\"]:.3f}')
print(f'Total scenarios: {len(data[\"eval_output_items\"])}')

for item in data['eval_output_items'][:3]:
    r = item['reasoning']
    print(f'  {item[\"id\"]}:')
    print(f'    F1={r[\"tool_selection_f1\"]:.2f}  predicted={r[\"predicted_tools\"]}')
    print(f'    expected={r[\"expected_tools\"]}')
"
```

**Example output:**
```
Average TSQ F1: 0.650
Total scenarios: 10
  banking_scenario_000:
    F1=1.00  predicted=['getaccountbalance']
    expected=['getaccountbalance']
  banking_scenario_001:
    F1=0.67  predicted=['getaccountbalance', 'gettransactionhistory']
    expected=['getaccountbalance', 'transferfunds']
  banking_scenario_002:
    F1=0.00  predicted=['scheduleappointment']
    expected=['getexchangerates']
```

### Score interpretation

| F1 Score | Meaning |
|----------|---------|
| 1.0 | All expected tools predicted, no extra tools |
| 0.5–0.9 | Partial match — some tools correct, some missing or extra |
| 0.0 | No overlap between predicted and expected tools |

---

## All Domains Evaluation

Download all 5 domains:

```bash
python examples/dynamo_integration/scripts/download_agent_leaderboard_v2.py \
  --output-dir data/agent_leaderboard \
  --domains banking healthcare insurance investment telecom
```

Run across all domains:

```bash
export AGENT_LEADERBOARD_DATA=data/agent_leaderboard/agent_leaderboard_v2_all.json
nat eval --config_file examples/benchmarks/agent_leaderboard/configs/eval_all_domains.yml
```

### Available domains

| Domain | Scenarios | Tools | Personas | Description |
|--------|-----------|-------|----------|-------------|
| `banking` | 100 | 20 | 100 | Account management, transfers, loans, cards |
| `healthcare` | 100 | 20 | 100 | Appointments, prescriptions, medical records |
| `insurance` | 100 | 20 | 100 | Policies, claims, coverage, renewals |
| `investment` | 100 | 20 | 100 | Portfolio management, stocks, trading |
| `telecom` | 100 | 20 | 100 | Plans, billing, support, device management |
| **Total** | **500** | **100** | **500** | |
