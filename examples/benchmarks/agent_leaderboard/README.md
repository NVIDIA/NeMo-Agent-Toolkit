<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

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
Average TSQ F1: 0.425
Total scenarios: 100
  banking_scenario_000:
    F1=0.67  predicted=['getaccountbalance', 'gettransactionhistory']
    expected=['getaccountbalance', 'transferfunds']
  banking_scenario_001:
    F1=1.00  predicted=['getloaninformation']
    expected=['getloaninformation']
  banking_scenario_002:
    F1=0.00  predicted=['scheduleappointment']
    expected=['getexchangerates']
```

### TSQ Score interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | All expected tools predicted, no extra tools |
| 0.5-0.9 | Partial match — some tools correct, some missing or extra |
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

| Domain | Description | Typical tool count |
|--------|-------------|-------------------|
| `banking` | Account management, transfers, loans | ~20 tools |
| `healthcare` | Appointments, prescriptions, records | ~15 tools |
| `insurance` | Policies, claims, coverage | ~15 tools |
| `investment` | Portfolio, stocks, trading | ~15 tools |
| `telecom` | Plans, billing, support | ~15 tools |
