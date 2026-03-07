<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# BYOB (Bring Your Own Benchmark) Evaluation

**Complexity:** 🟢 Beginner

Score NAT agent workflow outputs using [NeMo Evaluator's BYOB framework](https://docs.nvidia.com/nemo/evaluator/). BYOB lets you define custom benchmarks with any dataset and scorer — this example shows how to plug BYOB scorers into NAT's evaluation pipeline.

## Key Features

- **Any BYOB benchmark**: Use any benchmark defined with `@benchmark` + `@scorer` decorators
- **Built-in scorers**: `exact_match`, `contains`, `f1_token`, `bleu`, `rouge`, `regex_match`
- **Custom scorers**: Write your own scorer function with `ScorerInput`
- **No model_call_fn needed**: BYOB scorers run in-process — NAT handles the LLM calls
- **Dataset from benchmark**: Dataset path, prompt template, and target field come from the benchmark definition

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

| Scorer | Score field | Description |
|--------|-----------|-------------|
| `exact_match` | `correct` | Case-insensitive, whitespace-trimmed equality |
| `contains` | `correct` | Target is a substring of response |
| `f1_token` | `f1` | Token-level F1 (also returns `precision`, `recall`) |
| `regex_match` | `correct` | Target is a regex pattern matched against response |
| `bleu` | `bleu_1`..`bleu_4` | BLEU-1 through BLEU-4 with smoothing |
| `rouge` | `rouge_1`, `rouge_2`, `rouge_l` | ROUGE F1 scores |

Set `score_field` in the evaluator config to match your scorer:

```yaml
evaluators:
  byob:
    _type: byob_evaluator
    benchmark_module: /path/to/benchmark.py
    benchmark_name: my_benchmark
    score_field: f1  # For f1_token scorer
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
| `model_call_fn` | `Optional[Callable]` | Always `None` in NAT (LLM calls handled by workflow) |
| `config` | `Dict[str, Any]` | Benchmark's `extra` config |
