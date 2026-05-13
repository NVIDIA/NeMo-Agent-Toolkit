# Evaluator: `trajectory`

**Package:** `nvidia-nat[langchain]` — no extra install needed
**Best for:** Evaluating the quality of the agent's tool-call sequence (which tools it used, in what order, with what inputs)

## When to use

- Your agent uses one or more tools and you care about *how* it reached the answer, not just the final response
- You want to detect inefficiencies: wrong tool selection, unnecessary retries, skipped steps
- Multi-agent workflows where delegation decisions matter
- Error recovery scenarios: did the agent handle a tool failure gracefully?

## Config fields

The `trajectory` evaluator inherits from `EvaluatorLLMConfig`, which requires an `llm_name` pointing to a judge LLM.

| Field | Required | Description |
| --- | --- | --- |
| `llm_name` | Yes | Judge LLM from the `llms:` section. Must support reasoning about tool call sequences. |

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
    tool_usage:
      _type: trajectory
      llm_name: judge_llm
```

## What it evaluates

The evaluator receives the full `intermediate_steps` from the agent run — every LLM call and tool call, in order. The judge LLM scores the sequence on:

- Did the agent select the right tools for the task?
- Were tool inputs correct and well-formed?
- Was the sequence efficient (no unnecessary loops or retries)?
- Did the agent recover well if a tool failed?

## Output

Scores written to `output/tool_usage_output.json`. Each entry:

- `score` — 0 to 1 trajectory quality score
- `reasoning` — judge's explanation of what was good or bad about the tool sequence

## Dataset requirements

Your dataset entries benefit from an `expected_trajectory` field (optional) that describes the ideal tool sequence:

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris",
  "metadata": {
    "expected_tools": ["wikipedia_search"],
    "category": "factual"
  }
}
```

Without `expected_trajectory`, the judge evaluates purely on reasonableness of the sequence.

## Gotchas

- Requires `intermediate_steps` to be non-empty — if the agent never calls a tool, the trajectory evaluator returns a low score by default
- Use a capable judge model (≥70B); smaller models struggle to reason about multi-step sequences
- Works well combined with `tunable_rag_evaluator` or `langsmith_judge` to cover both trajectory quality and response quality
