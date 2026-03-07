<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# BFCL Benchmark Evaluation

**Complexity:** 🟡 Intermediate

Evaluate NAT agent workflows against the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) v3 benchmark. BFCL tests single-turn function calling accuracy across simple, parallel, and multiple function call scenarios.

This example supports **three evaluation modes** that demonstrate different NAT integration patterns:

| Mode | Workflow Type | How it works |
|------|--------------|--------------|
| **AST Prompting** | `bfcl_ast_workflow` | Function schemas in system prompt; LLM outputs raw function call text |
| **Native FC** | `bfcl_fc_workflow` | `llm.bind_tools(schemas)` + `ainvoke()`; extracts structured `tool_calls` |
| **ReAct** | `bfcl_react_workflow` | Multi-step tool-calling loop with stub responses and intent capture |

## Table of Contents

- [Installation](#installation)
- [Set Up Environment](#set-up-environment)
- [Evaluation Mode 1: AST Prompting](#evaluation-mode-1-ast-prompting)
- [Evaluation Mode 2: Native Function Calling](#evaluation-mode-2-native-function-calling)
- [Evaluation Mode 3: ReAct Loop](#evaluation-mode-3-react-loop)
- [Understanding Results](#understanding-results)
- [Test Categories](#test-categories)

---

## Installation

```bash
uv pip install -e examples/benchmarks/bfcl
```

### Prerequisites

The NVIDIA `bfcl` package must be installed (includes datasets and AST checker):

```bash
pip install nvidia-bfcl
```

---

## Set Up Environment

```bash
export NVIDIA_API_KEY=<your-nvidia-api-key>

# Locate the BFCL dataset file
python -c "
from bfcl.constant import PROMPT_PATH
print(f'export BFCL_DATASET_FILE={PROMPT_PATH}/BFCL_v3_simple.json')
"
```

```bash
export BFCL_DATASET_FILE=<path from above>
```

---

## Evaluation Mode 1: AST Prompting

The LLM receives function schemas as JSON text in the system prompt and outputs raw function call text like `calculate_area(base=10, height=5)`.

```bash
nat eval --config_file examples/benchmarks/bfcl/configs/eval_ast_simple.yml
```

**Expected output:**
```
INFO - Starting evaluation run with config file: .../eval_ast_simple.yml
INFO - Loaded 3 possible answers from .../possible_answer/BFCL_v3_simple.json
INFO - Loaded 100 BFCL entries from BFCL_v3_simple.json (category: simple)
Running workflow: 100%|██████████| 100/100 [02:30<00:00, 1.50s/it]
INFO - BFCL evaluation complete: accuracy=0.850 (85/100) category=simple

=== EVALUATION SUMMARY ===
| Evaluator |   Avg Score | Output File      |
|-----------|-------------|------------------|
| bfcl      |       0.850 | bfcl_output.json |
```

> The system prompt uses BFCL's standard format instruction which constrains the LLM to output
> `[func_name(param=value)]` format. Accuracy varies by model — `llama-3.3-70b-instruct`
> typically scores 80-95% on the simple split.

---

## Evaluation Mode 2: Native Function Calling

Uses `llm.bind_tools(schemas)` — the LLM makes structured tool calls via the native `tools=` API parameter. Tool call args are extracted from `AIMessage.tool_calls` and formatted for BFCL scoring.

```bash
nat eval --config_file examples/benchmarks/bfcl/configs/eval_fc_simple.yml
```

This mode typically achieves higher accuracy than AST prompting because the LLM uses its native function calling capability rather than generating text.

---

## Evaluation Mode 3: ReAct Loop

Drives a multi-step reasoning loop: the LLM reasons about which tool to call, executes it (stub returns a canned response), observes the result, and decides whether to call more tools. Tool call intents are captured and deduplicated for scoring.

```bash
nat eval --config_file examples/benchmarks/bfcl/configs/eval_react_simple.yml
```

This mode demonstrates NAT-native agent execution against BFCL — the agent can reason step-by-step before making tool calls.

---

## Understanding Results

Results are saved to `.tmp/nat/benchmarks/bfcl/<mode>_simple/`:

```bash
python -c "
import json
with open('.tmp/nat/benchmarks/bfcl/ast_simple/bfcl_output.json') as f:
    data = json.load(f)
print(f'Accuracy: {data[\"average_score\"]:.3f}')

# Show first few results
for item in data['eval_output_items'][:5]:
    status = 'PASS' if item['score'] == 1.0 else 'FAIL'
    print(f'  {item[\"id\"]}: {status}')
    if item['score'] == 0.0:
        r = item.get('reasoning', {})
        if 'error' in r:
            print(f'    Error: {r[\"error\"][:100]}')
"
```

**Example output:**
```
Accuracy: 0.850
  simple_0: PASS
  simple_1: PASS
  simple_2: PASS
  simple_3: FAIL
    Error: ["Incorrect type for parameter 'x'. Expected type integer, got str."]
  simple_4: PASS
```

### Score interpretation

- **1.0**: Function name, parameter names, types, and values all match ground truth
- **0.0**: Any mismatch — wrong function, wrong params, wrong types, or unparseable output

---

## Test Categories

Change the `test_category` and `file_path` in the config to evaluate different BFCL splits:

| Category | File | Description |
|----------|------|-------------|
| `simple` | `BFCL_v3_simple.json` | Single function call |
| `multiple` | `BFCL_v3_multiple.json` | Select one of multiple candidate functions |
| `parallel` | `BFCL_v3_parallel.json` | Multiple function calls in one response |
| `parallel_multiple` | `BFCL_v3_parallel_multiple.json` | Combined parallel + multiple |
| `java` | `BFCL_v3_java.json` | Java function call syntax |
| `javascript` | `BFCL_v3_javascript.json` | JavaScript function call syntax |
| `irrelevance` | `BFCL_v3_irrelevance.json` | Model should refuse (no valid tool) |

For Java/JavaScript categories, set `language` in the evaluator config:

```yaml
evaluators:
  bfcl:
    _type: bfcl_evaluator
    test_category: java
    language: Java
```
