# Evaluator: `ragas`

**Package:** `nvidia-nat[ragas]` — **separate extra, not included in `[langchain]`**
**Best for:** Standard RAG evaluation metrics (AnswerAccuracy, ResponseGroundedness, ContextRelevance) backed by the RAGAS library

## Installation

The `ragas` evaluator is a **separate optional extra**. It is NOT included when you install `nvidia-nat[langchain]`. You must explicitly add it:

```toml
# pyproject.toml
dependencies = [
    "nvidia-nat[langchain,ragas]>=1.6.0",
]
```

Or add it to an existing project:

```bash
uv add "nvidia-nat[ragas]>=1.6.0"
```

If you see `ModuleNotFoundError` or `nat info components -t evaluator` does not list `ragas`, the extra is not installed.

**If you cannot install the extra**, use `tunable_rag_evaluator` instead — it covers the same RAG quality dimensions (coverage ≈ AnswerAccuracy, relevance ≈ ContextRelevance) and is always available with `[langchain]`. See `evaluator-tunable-rag.md`.

## When to use

- You need standardized RAGAS metrics for benchmark comparability
- Your team already uses RAGAS elsewhere and wants consistent metrics
- You need `ContextPrecision` or `ContextRecall` (not available in `tunable_rag_evaluator`)

## Core metrics

| Metric | What it measures | Reference needed? |
| --- | --- | --- |
| `AnswerAccuracy` | Semantic accuracy vs. reference answer | Yes |
| `FactualCorrectness` | Factual accuracy (accepts `mode` kwarg for strictness) | Yes |
| `ResponseGroundedness` | Response supported by retrieved context | No |
| `ContextRelevance` | Retrieved context relevant to the question | No |
| `Faithfulness` | Claims inferable from given context (stricter than Groundedness) | No |
| `AnswerRelevancy` | Whether the answer addresses the question | No |
| `ContextPrecision` | Proportion of retrieved context that is relevant | Yes |
| `ContextRecall` | Whether all relevant info was retrieved | Yes |
| `NoiseSensitivity` | Robustness to irrelevant context | Yes |

Available metrics depend on the installed `ragas` package version. Check [docs.ragas.io](https://docs.ragas.io) for the full list.

## Example

```yaml
llms:
  judge_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct

eval:
  general:
    dataset:
      _type: json
      file_path: data/dataset.json
  evaluators:
    answer_accuracy:
      _type: ragas
      metric: AnswerAccuracy
      llm_name: judge_llm

    groundedness:
      _type: ragas
      metric: ResponseGroundedness
      llm_name: judge_llm

    context_relevance:
      _type: ragas
      metric: ContextRelevance
      llm_name: judge_llm
```

## Gotchas

- **Must install `nvidia-nat[ragas]`** — attempting to use `_type: ragas` without it will fail at eval runtime, not at config validation
- Each metric is a separate evaluator entry — one `_type: ragas` block per metric
- RAGAS metrics that require context (`ResponseGroundedness`, `ContextRelevance`) need the agent to expose its retrieved context in the output. If the agent doesn't surface context in `intermediate_steps`, these metrics may return 0
- Metric names are case-sensitive — use exact names from the RAGAS docs
- Some metrics accept kwargs — pass them as a dict: `metric: {FactualCorrectness: {mode: precision}}`
