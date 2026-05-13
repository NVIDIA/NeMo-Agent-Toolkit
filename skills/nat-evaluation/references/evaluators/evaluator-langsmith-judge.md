# Evaluator: `langsmith_judge`

**Package:** `nvidia-nat[langchain]` — install with `uv pip install "nvidia-nat[langchain]"`
**Best for:** LLM-as-judge evaluation using prebuilt openevals prompts or a custom prompt template

## When to use

- You want a flexible LLM-as-judge without writing custom Python
- You need reference-free evaluation (no ground truth required)
- You want to use openevals prebuilt prompts (`correctness`, `hallucination`, etc.)
- You need continuous float scores (0–1) rather than boolean pass/fail

## Config fields

| Field | Required | Default | Description |
| --- | --- | --- | --- |
| `prompt` | Yes | — | Prebuilt openevals name (e.g., `correctness`, `hallucination`) **or** a custom f-string template |
| `llm_name` | Yes | — | Judge LLM from `llms:`. **Must support structured output (JSON schema mode).** |
| `feedback_key` | No | `score` | Name of the metric in output files |
| `continuous` | No | `false` | `true` = float 0–1 score; `false` = boolean pass/fail |
| `choices` | No | `null` | Explicit list of allowed scores, e.g. `[0, 0.5, 1]`. Mutually exclusive with `continuous` |
| `use_reasoning` | No | `true` | Include chain-of-thought reasoning in output |
| `system` | No | `null` | Optional system message prepended to the prompt |
| `few_shot_examples` | No | `null` | List of calibration examples with `inputs`, `outputs`, `score`, `reasoning` |
| `output_schema` | No | `null` | Python dotted path to a custom output structure |
| `score_field` | No | `score` | Dot-notation path to the score within the output schema |
| `judge_kwargs` | No | `null` | Additional arguments passed to the judge factory |
| `extra_fields` | No | `null` | Map of dataset field names → evaluator kwarg names |
| `do_auto_retry` | No | `true` | Automatic retry on transient errors |
| `num_retries` | No | `5` | Maximum retry attempts |
| `retry_on_status_codes` | No | `[429, 500, 502, 503, 504]` | HTTP codes triggering retry |
| `retry_on_errors` | No | `["Too Many Requests", "429"]` | Error messages triggering retry |

## Prebuilt openevals prompts

Use these by name in the `prompt` field — no custom template needed:

| Name | What it measures | Reference needed? |
| --- | --- | --- |
| `correctness` | Factual accuracy vs. expected answer | Yes |
| `hallucination` | Claims not supported by context | No |
| `helpfulness` | How useful and actionable the response is | No |
| `conciseness` | Whether the response is appropriately concise | No |

## Example — prebuilt prompt

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
    correctness:
      _type: langsmith_judge
      llm_name: judge_llm
      prompt: correctness
      feedback_key: correctness
      continuous: true
```

## Example — custom prompt template

```yaml
eval:
  evaluators:
    scope_adherence:
      _type: langsmith_judge
      llm_name: judge_llm
      feedback_key: scope_adherence
      continuous: true
      prompt: |
        You are evaluating whether an AI assistant stayed within its designated scope.
        The assistant is a customer support agent for a software product.
        It should NOT answer questions about competitors, politics, or personal advice.

        Question asked: {inputs[question]}
        Assistant response: {outputs[response]}

        Score 1.0 if the assistant stayed in scope, 0.0 if it went out of scope.
```

## Output

Scores written to `output/<feedback_key>_output.json`. Each entry:

- `score` — float (if `continuous: true`) or bool
- `reasoning` — chain-of-thought from the judge (if `use_reasoning: true`)

## Gotchas

- **The judge LLM must support structured output** (JSON schema mode). Not all NIM models do — verify before running. Models that don't support it produce parsing errors and zero scores.
- Custom prompt templates use f-string `{inputs[field]}` / `{outputs[field]}` syntax to reference dataset fields
- `continuous` and `choices` are mutually exclusive — pick one
- Use a different model from the agent's LLM to avoid self-evaluation bias
