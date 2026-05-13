# Evaluator: `tunable_rag_evaluator`

**Package:** `nvidia-nat[langchain]` — no extra install needed
**Best for:** RAG agents where you want customizable scoring across coverage, correctness, and relevance

## When to use

- Your agent retrieves context and generates a response based on it
- You need a single evaluator that produces multiple sub-scores (coverage, correctness, relevance)
- You want to tune the relative weight of those sub-scores without writing custom Python
- Fallback when `ragas` is not installed

## Config fields

| Field | Required | Description |
| --- | --- | --- |
| `llm_name` | Yes | Judge LLM from the `llms:` section. Must be a different model from the agent's LLM. |
| `judge_llm_prompt` | Yes | Prompt instructing the judge. Leave blank to use the built-in default. |
| `default_scoring` | No (default: `false`) | Use the built-in coverage/correctness/relevance prompt. Set to `true` to skip writing your own prompt. |
| `default_score_weights` | No | Weights for the three sub-scores when `default_scoring: true`. Default: `coverage: 0.5, correctness: 0.3, relevance: 0.2` |
| `llm_retry_control_params` | No | Retry configuration object (see below) |

**Retry configuration** (`llm_retry_control_params`):

| Field | Default | Description |
| --- | --- | --- |
| `stop_after_attempt` | — | Maximum retry attempts |
| `initial_backoff_delay_seconds` | — | Initial delay between retries |
| `has_exponential_jitter` | — | Add randomized exponential backoff |

## Minimal example (default scoring)

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
    rag_quality:
      _type: tunable_rag_evaluator
      llm_name: judge_llm
      judge_llm_prompt: ""        # ignored when default_scoring is true
      default_scoring: true
      default_score_weights:
        coverage: 0.5
        correctness: 0.3
        relevance: 0.2
```

## Custom scoring example

```yaml
eval:
  evaluators:
    rag_quality:
      _type: tunable_rag_evaluator
      llm_name: judge_llm
      default_scoring: false
      judge_llm_prompt: |
        You are evaluating a research assistant. Given the question, expected answer,
        and generated answer, score the response from 0 to 1 on how well it answers
        the question with accurate, complete information.
        Question: {question}
        Expected: {answer}
        Generated: {generated_answer}
        Return a JSON with keys: score (float 0-1), reasoning (string).
```

## Output

Scores are written to `output/rag_quality_output.json`. Each entry contains:

- `score` — weighted composite (0–1)
- `reasoning` — judge's explanation
- Per-dimension sub-scores when using `default_scoring: true`

## Gotchas

- `judge_llm_prompt` is **required** in the config even when `default_scoring: true` — pass an empty string `""`
- The judge LLM must be a separate model from the agent's LLM to avoid self-evaluation bias
- Sub-scores vary by judge LLM capability — use a model ≥70B parameters for reliable scoring
