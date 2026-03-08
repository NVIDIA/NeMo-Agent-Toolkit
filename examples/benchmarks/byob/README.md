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

# BYOB (Bring Your Own Benchmark) Evaluation

**Complexity:** 🟡 Intermediate

Run [NeMo Evaluator BYOB](https://docs.nvidia.com/nemo/evaluator/latest/) benchmarks directly on NeMo Agent Toolkit workflows — without re-implementing the dataset loader or scorer logic.

## Why BYOB in NeMo Agent Toolkit?

The NeMo Evaluator BYOB (Bring Your Own Benchmark) framework lets users define custom evaluation benchmarks using the `@benchmark` and `@scorer` decorators. A benchmark definition specifies a dataset, a prompt template, and a scorer function — everything needed to evaluate a model.

Normally, NeMo Evaluator runs the full pipeline: it loads the dataset, renders prompts, calls the model endpoint, and scores the responses. **This integration reuses the benchmark definition, dataset loading, and scorer functions from NeMo Evaluator, but replaces the model-calling step with NeMo Agent Toolkit workflow execution.** This means:

- **Existing BYOB benchmarks work as-is** — the same `@benchmark` + `@scorer` Python file you use with NeMo Evaluator works identically with NeMo Agent Toolkit. No re-implementation needed.
- **NeMo Agent Toolkit handles the agent execution** — instead of calling a model endpoint, the toolkit runs its own workflow (tool-calling agents, RAG pipelines, multi-step reasoning chains) to generate responses.
- **Scorers run in-process** — the scorer receives `ScorerInput(response=workflow_output, target=ground_truth)` and returns a score dict. `model_call_fn` is `None` (NeMo Agent Toolkit handles all LLM calls upstream).

```
┌─────────────────────────────────────┐
│  BYOB Benchmark Definition (.py)    │  ← Written once, used in both systems
│  @benchmark + @scorer               │
├───────────────┬─────────────────────┤
│ NeMo Evaluator│  NAT Integration    │
│ (standalone)  │  (this example)     │
│               │                     │
│ load_dataset()│  load_dataset()     │  ← Same function, reused
│ render_prompt │  NAT workflow       │  ← NAT replaces model calling
│ call_model()  │  (agents, tools)    │
│ scorer_fn()   │  scorer_fn()        │  ← Same scorer, reused
└───────────────┴─────────────────────┘
```

## Key Features

- **Direct reuse**: Any benchmark defined with `@benchmark` + `@scorer` works without modification
- **Built-in scorers**: `exact_match`, `contains`, `f1_token`, `bleu`, `rouge`, `regex_match` — all from `nemo_evaluator.contrib.byob.scorers`
- **Custom scorers**: Write your own scorer function with `ScorerInput`
- **HuggingFace datasets**: The BYOB `load_dataset()` function supports `hf://` URIs, local JSONL, CSV, and TSV
- **Dataset from benchmark**: Dataset path, prompt template, and target field come from the benchmark definition — no duplication in the eval config

## Table of Contents

- [Installation](#installation)
- [Step 1: Define a Benchmark](#step-1-define-a-benchmark)
- [Step 2: Configure the Evaluation](#step-2-configure-the-evaluation)
- [Step 3: Run the Evaluation](#step-3-run-the-evaluation)
- [Understanding Results](#understanding-results)
- [Built-in Scorers Reference](#built-in-scorers-reference)
- [Writing a Custom Scorer](#writing-a-custom-scorer)

---

## Installation

```bash
uv pip install -e examples/benchmarks/byob
```

### Prerequisites

NeMo Evaluator must be installed for BYOB support:

```bash
pip install nemo-evaluator
```

---

## Step 1: Define a Benchmark

Create a Python file with your benchmark definition. Here's a minimal example using `exact_match`:

```python
# my_benchmark.py
import json, os, tempfile
from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer
from nemo_evaluator.contrib.byob.scorers import exact_match

# Create a simple QA dataset
DATA = [
    {"id": "0", "question": "What is 2+2?", "target": "4"},
    {"id": "1", "question": "Capital of France?", "target": "Paris"},
    {"id": "2", "question": "Color of the sky?", "target": "blue"},
]
DATASET_PATH = os.path.join(tempfile.gettempdir(), "my_qa_dataset.jsonl")
with open(DATASET_PATH, "w") as f:
    for row in DATA:
        f.write(json.dumps(row) + "\n")

@benchmark(
    name="my-qa-test",
    dataset=DATASET_PATH,
    prompt="{question}",
    target_field="target",
)
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
```

---

## Step 2: Configure the Evaluation

Set the benchmark module path:

```bash
export BYOB_BENCHMARK_MODULE=/path/to/my_benchmark.py
export BYOB_BENCHMARK_NAME=my_qa_test    # Normalized: hyphens become underscores
export NVIDIA_API_KEY=<your-nvidia-api-key>
```

---

## Step 3: Run the Evaluation

```bash
nat eval --config_file examples/benchmarks/byob/configs/eval_exact_match.yml
```

**Expected output:**
```
INFO - Starting evaluation run with config file: .../eval_exact_match.yml
INFO - Imported BYOB benchmark 'my-qa-test' (dataset: /tmp/my_qa_dataset.jsonl)
INFO - Loaded 3 BYOB samples from benchmark 'my-qa-test'
Running workflow: 100%|██████████| 3/3 [00:15<00:00, 5.00s/it]
INFO - BYOB evaluation complete: avg_correct=0.XXX (3 items)

=== EVALUATION SUMMARY ===
| Evaluator |   Avg Score | Output File      |
|-----------|-------------|------------------|
| byob      |       0.XXX | byob_output.json |
```

---

## Understanding Results

### The `byob_evaluator`

This example uses the **`byob_evaluator`** (`_type: byob_evaluator` in the eval config). It imports the benchmark definition at evaluation time, then calls `bench.scorer_fn(ScorerInput(...))` for each item to produce a score dict.

The evaluator is configured in the YAML under `eval.evaluators`:

```yaml
evaluators:
  byob:
    _type: byob_evaluator
    benchmark_module: /path/to/my_benchmark.py   # Same file used for the dataset
    benchmark_name: my_qa_test                    # Normalized benchmark name
    score_field: correct                          # Key from scorer output to use as primary score
```

The `score_field` parameter controls which key from the scorer's output dict becomes the item's primary score. For `exact_match` and `contains`, this is `correct` (boolean → 1.0 or 0.0). For `f1_token`, you'd set `score_field: f1` to get the F1 float. The `average_score` in the output is the mean of all items' primary scores.

The metrics available in each item's `reasoning` dict depend entirely on what the scorer function returns — the evaluator passes through the full scorer output without modification.

### Inspect results

```bash
python -c "
import json
with open('.tmp/nat/benchmarks/byob/exact_match/byob_output.json') as f:
    data = json.load(f)
print(f'Average score: {data[\"average_score\"]:.3f}')
for item in data['eval_output_items']:
    r = item['reasoning']
    print(f'  {item[\"id\"]}: score={item[\"score\"]} correct={r.get(\"correct\", \"N/A\")}')
"
```

**Example output:**
```
Average score: 0.667
  0: score=1.0 correct=True
  1: score=1.0 correct=True
  2: score=0.0 correct=False
```

---

## Built-in Scorers Reference

BYOB scorers come from the [NeMo Evaluator](https://docs.nvidia.com/nemo/evaluator/latest/) framework (`nemo_evaluator.contrib.byob.scorers`). They are standard NLP evaluation metrics packaged as simple Python functions that accept a `ScorerInput` and return a dict of metric values. When you use BYOB through NeMo Agent Toolkit, the scorer runs in-process against the workflow output — the toolkit handles the LLM calls and the scorer only sees the final `(response, target)` pair.

You can use any built-in scorer directly, compose them with `any_of()` / `all_of()`, or write your own from scratch. The only requirement is the signature: `def scorer(sample: ScorerInput) -> dict`.

### Simple scorers

These return a single `correct` boolean. Use `score_field: correct` in the evaluator config.

| Scorer | Import | What it checks |
|--------|--------|----------------|
| `exact_match` | `from nemo_evaluator.contrib.byob.scorers import exact_match` | Case-insensitive, whitespace-trimmed string equality |
| `contains` | `from nemo_evaluator.contrib.byob.scorers import contains` | Whether the target appears as a substring of the response |
| `regex_match` | `from nemo_evaluator.contrib.byob.scorers import regex_match` | Whether the response matches a regex pattern (target is the pattern) |

### Metric scorers

These return numeric scores. Set `score_field` to the metric you want as the primary score.

| Scorer | Import | Returns |
|--------|--------|---------|
| `f1_token` | `from nemo_evaluator.contrib.byob.scorers import f1_token` | `{"f1": float, "precision": float, "recall": float}` — token-level overlap |
| `bleu` | `from nemo_evaluator.contrib.byob.scorers import bleu` | `{"bleu_1": float, ..., "bleu_4": float}` — sentence-level BLEU with smoothing |
| `rouge` | `from nemo_evaluator.contrib.byob.scorers import rouge` | `{"rouge_1": float, "rouge_2": float, "rouge_l": float}` — ROUGE F1 scores |

### Composer functions

Combine multiple scorers into one:

| Function | What it does |
|----------|-------------|
| `any_of(exact_match, contains)` | Correct if **any** scorer passes |
| `all_of(exact_match, regex_match)` | Correct only if **all** scorers pass |

```python
from nemo_evaluator.contrib.byob.scorers import exact_match, contains, any_of

@scorer
def flexible_scorer(sample: ScorerInput) -> dict:
    return any_of(exact_match, contains)(sample)
```

### Configuring `score_field`

Set `score_field` in the evaluator config to select which key from the scorer output dict to use as the primary score:

```yaml
evaluators:
  byob:
    _type: byob_evaluator
    benchmark_module: /path/to/benchmark.py
    benchmark_name: my_benchmark
    score_field: f1  # For f1_token scorer (default: "correct")
```

---

## Writing a Custom Scorer

```python
from nemo_evaluator.contrib.byob import ScorerInput, benchmark, scorer

@benchmark(
    name="my-custom-benchmark",
    dataset="hf://squad",  # HuggingFace dataset
    prompt="Context: {context}\nQuestion: {question}\nAnswer:",
    target_field="answers",
)
@scorer
def custom_scorer(sample: ScorerInput) -> dict:
    """Score based on whether any valid answer appears in the response."""
    response = sample.response.lower()
    target = sample.target

    # Handle SQuAD-style answers (list of valid answers)
    if isinstance(target, dict) and "text" in target:
        valid_answers = [a.lower() for a in target["text"]]
    elif isinstance(target, list):
        valid_answers = [str(a).lower() for a in target]
    else:
        valid_answers = [str(target).lower()]

    correct = any(answer in response for answer in valid_answers)
    return {"correct": correct}
```

### ScorerInput fields

| Field | Type | Description |
|-------|------|-------------|
| `response` | `str` | Model's generated response |
| `target` | `Any` | Ground-truth value from dataset |
| `metadata` | `dict` | Full dataset row |
| `model_call_fn` | `Optional[Callable]` | Always `None` in NeMo Agent Toolkit (LLM calls handled by workflow) |
| `config` | `Dict[str, Any]` | Benchmark's `extra` config |
