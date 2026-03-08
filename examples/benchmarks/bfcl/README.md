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

# BFCL Benchmark Evaluation

**Complexity:** 🟡 Intermediate

Evaluate NeMo Agent Toolkit agent workflows against the [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/leaderboard.html) v3 benchmark. BFCL tests single-turn function calling accuracy across simple, parallel, and multiple function call scenarios.

This example supports **three evaluation modes** that demonstrate different NeMo Agent Toolkit integration patterns:

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

The NVIDIA `nvidia-bfcl` package must be installed (includes datasets and AST checker).
Due to an overly restrictive `numpy==1.26.4` pin in `nvidia-bfcl` that conflicts with
NeMo Agent Toolkit's `numpy>=2.3`, install it with `--no-deps`:

```bash
uv pip install nvidia-bfcl --no-deps
```

> `nvidia-bfcl` works correctly with NumPy 2.x at runtime — the pin is a packaging
> constraint only. The `[bfcl]` extra in `nvidia-nat-benchmarks` installs the required
> `tree-sitter` dependencies automatically.

---

## Set Up Environment

### 1. Set your NVIDIA API key

```bash
export NVIDIA_API_KEY=<your-nvidia-api-key>
```

Or add it to a `.env` file in your project root (NeMo Agent Toolkit loads `.env` automatically).

### 2. Locate the BFCL dataset

The `nvidia-bfcl` package ships with all BFCL v3 datasets and ground-truth answers as JSONL files. Each line is one test entry containing a user question and one or more function schemas. The ground-truth answers (in `possible_answer/`) contain the expected function call(s) with parameter values.

**Dataset structure:**

```
bfcl/data/
├── BFCL_v3_simple.json                  # 400 single function call test entries
├── BFCL_v3_multiple.json                # 200 select-one-of-many function entries
├── BFCL_v3_parallel.json                # 200 multiple calls in one response
├── BFCL_v3_parallel_multiple.json       # 200 combined parallel + multiple
├── BFCL_v3_java.json                    # 100 Java function call syntax
├── BFCL_v3_javascript.json              #  50 JavaScript function call syntax
├── BFCL_v3_irrelevance.json             # 240 model should refuse (no valid tool)
├── BFCL_v3_live_simple.json             # 258 live/real-world simple calls
├── BFCL_v3_live_multiple.json           # 1053 live multiple function entries
├── BFCL_v3_sql.json                     # 100 SQL function calls
└── possible_answer/
    ├── BFCL_v3_simple.json              # Ground-truth answers for simple
    ├── BFCL_v3_multiple.json            # Ground-truth answers for multiple
    └── ...                              # (not all categories have answers)
```

Each test entry looks like:

```json
{
  "id": "simple_0",
  "question": [[{"role": "user", "content": "Find the area of a triangle with base 10 and height 5."}]],
  "function": [{"name": "calculate_triangle_area", "description": "...", "parameters": {...}}]
}
```

The matching ground-truth answer:

```json
{
  "id": "simple_0",
  "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
}
```

**Locate the installed dataset path:**

```bash
python -c "
from bfcl.constant import PROMPT_PATH, POSSIBLE_ANSWER_PATH
print(f'Dataset dir:  {PROMPT_PATH}')
print(f'Answers dir:  {POSSIBLE_ANSWER_PATH}')
"
```

**Expected output:**
```
Dataset dir:  /path/to/.venv/lib/python3.11/site-packages/bfcl/data
Answers dir:  /path/to/.venv/lib/python3.11/site-packages/bfcl/data/possible_answer
```

### 3. Set the dataset path

Point `BFCL_DATASET_FILE` to the specific test category you want to evaluate:

```bash
# Simple (400 entries) — recommended starting point
export BFCL_DATASET_FILE=/path/to/.venv/lib/python3.11/site-packages/bfcl/data/BFCL_v3_simple.json
```

> The evaluator automatically resolves the matching `possible_answer/` file from the same directory. If no answer file exists for a category, entries score 0 (the evaluator needs ground truth to compare against).

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

> The system prompt uses the standard BFCL format instruction which constrains the LLM to output
> `[func_name(param=value)]` format. Accuracy varies by model — `llama-3.3-70b-instruct`
> typically scores 80-95% on the simple split.

---

## Evaluation Mode 2: Native Function Calling

Uses `llm.bind_tools(schemas)` — the LLM makes structured tool calls via the native `tools=` API parameter. Tool call arguments are extracted from `AIMessage.tool_calls` and formatted for BFCL scoring.

```bash
nat eval --config_file examples/benchmarks/bfcl/configs/eval_fc_simple.yml
```

**Expected output:**
```
INFO - Starting evaluation run with config file: .../eval_fc_simple.yml
INFO - Loaded 400 possible answers from .../possible_answer/BFCL_v3_simple.json
INFO - Loaded 400 BFCL entries from BFCL_v3_simple.json (category: simple)
Running workflow: 100%|██████████| 400/400 [05:30<00:00, 1.21s/it]
INFO - BFCL evaluation complete: accuracy=0.720 (288/400) category=simple

=== EVALUATION SUMMARY ===
| Evaluator |   Avg Score | Output File      |
|-----------|-------------|------------------|
| bfcl      |       0.720 | bfcl_output.json |
```

> FC mode converts BFCL function schemas to OpenAI tool format and uses `bind_tools()`.
> The LLM returns structured `tool_calls` with typed arguments, but accuracy can be lower
> than AST prompting when the schema type conversion (BFCL types → OpenAPI types) loses
> precision or the model returns string values where integers are expected.

---

## Evaluation Mode 3: ReAct Loop

Drives a multi-step reasoning loop: the LLM reasons about which tool to call, executes it (stub returns a canned response), observes the result, and decides whether to call more tools. Tool call intents are captured and deduplicated for scoring.

```bash
nat eval --config_file examples/benchmarks/bfcl/configs/eval_react_simple.yml
```

**Expected output:**
```
INFO - Starting evaluation run with config file: .../eval_react_simple.yml
INFO - Loaded 400 possible answers from .../possible_answer/BFCL_v3_simple.json
INFO - Loaded 400 BFCL entries from BFCL_v3_simple.json (category: simple)
Running workflow: 100%|██████████| 400/400 [12:00<00:00, 1.80s/it]
INFO - BFCL evaluation complete: accuracy=0.900 (360/400) category=simple

=== EVALUATION SUMMARY ===
| Evaluator |   Avg Score | Output File      |
|-----------|-------------|------------------|
| bfcl      |       0.900 | bfcl_output.json |
```

> ReAct mode uses `bind_tools()` like FC mode but adds a multi-step reasoning loop with
> canned tool responses. It also applies type coercion (string → int/float based on the BFCL
> schema) and deduplication of repeated tool calls, which typically produces the highest
> accuracy of the three modes. The tradeoff is higher latency due to multiple LLM calls per item.

---

## Understanding Results

### The `bfcl_evaluator`

All three evaluation modes use the same evaluator: **`bfcl_evaluator`** (`_type: bfcl_evaluator` in the eval config). It calls the BFCL `ast_checker()` function directly in Python to validate the model's function call output against the ground-truth possible answers. The evaluator handles three steps:

1. **Extract** the function call from the model's raw output (strips markdown, prose, JSON wrapping)
2. **Decode** the extracted text into a structured function call via the BFCL `default_decode_ast_prompting()` utility
3. **Check** the decoded call against ground truth via `ast_checker()` — validates function name, parameter names, types, and values

The evaluator is configured in the YAML under `eval.evaluators`:

```yaml
evaluators:
  bfcl:
    _type: bfcl_evaluator
    test_category: simple      # Must match the dataset's test category
    language: Python           # Python, Java, or JavaScript
```

Each item scores **1.0** (all checks pass) or **0.0** (any mismatch). The `average_score` is the accuracy across all items.

### Output files

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
- **0.0**: Any mismatch — wrong function, wrong parameters, wrong types, or output that cannot be parsed

---

## Test Categories

Change the `test_category` and `file_path` in the config to evaluate different BFCL splits:

| Category | File | Samples | Has Answers | Description |
|----------|------|---------|-------------|-------------|
| `simple` | `BFCL_v3_simple.json` | 400 | Yes | Single function call |
| `multiple` | `BFCL_v3_multiple.json` | 200 | Yes | Select one of multiple candidate functions |
| `parallel` | `BFCL_v3_parallel.json` | 200 | Yes | Multiple function calls in one response |
| `parallel_multiple` | `BFCL_v3_parallel_multiple.json` | 200 | Yes | Combined parallel + multiple |
| `java` | `BFCL_v3_java.json` | 100 | Yes | Java function call syntax |
| `javascript` | `BFCL_v3_javascript.json` | 50 | Yes | JavaScript function call syntax |
| `irrelevance` | `BFCL_v3_irrelevance.json` | 240 | No | Model should refuse (no valid tool) |
| `live_simple` | `BFCL_v3_live_simple.json` | 258 | Yes | Real-world simple function calls |
| `live_multiple` | `BFCL_v3_live_multiple.json` | 1,053 | Yes | Real-world multiple function calls |
| `sql` | `BFCL_v3_sql.json` | 100 | Yes | SQL function calls |

> Categories without answers (`Has Answers: No`) can still run the workflow, but the evaluator
> cannot score them — all entries will score 0. The `irrelevance` category is the exception:
> the evaluator checks that the model does *not* produce a function call (no answer file needed).

For Java/JavaScript categories, set `language` in the evaluator config:

```yaml
evaluators:
  bfcl:
    _type: bfcl_evaluator
    test_category: java
    language: Java
```
