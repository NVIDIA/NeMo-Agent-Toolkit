# LLM Configuration

How to wire NVIDIA Inference (hosted), NVIDIA NIM (local/self-hosted), and multi-LLM setups for production agents.

## NVIDIA Inference API (internal, hosted)

Use `_type: openai` with the NVIDIA inference base URL and `$NVIDIA_INFERENCE_API_KEY` as the api key:

```yaml
llms:
  nvidia_llm:
    _type: openai
    model_name: aws/anthropic/claude-haiku-4-5-v1
    base_url: https://inference-api.nvidia.com/v1
    temperature: 0.0
    api_key: $NVIDIA_INFERENCE_API_KEY
```

Then reference `nvidia_llm` as the `llm_name` in the workflow section.

## NVIDIA NIM (local or self-hosted)

Use `_type: nim` when connecting to a locally deployed NIM container or self-hosted NIM endpoint. This is the most common provider in production deployments.

```yaml
llms:
  instruct_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct
    temperature: 0.0
    base_url: ${INSTRUCT_LLM_BASE_URL:-http://nim-llm:8000/v1}
    max_tokens: 20000
    api_key: not-needed    # local NIM doesn't require an API key
```

## Multiple LLMs for Different Roles

Production agents typically define multiple LLMs — one for fast instruction-following, one for reasoning, and optionally one for evaluation:

```yaml
llms:
  instruct_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct
    temperature: 0.0
    base_url: http://instruct-llm:8000/v1
    max_tokens: 20000
    api_key: not-needed

  reasoning_llm:
    _type: nim
    model_name: nvidia/llama-3.3-nemotron-super-49b-v1.5
    temperature: 0.5
    base_url: http://reasoning-llm:8000/v1
    max_tokens: 5000
    api_key: not-needed

  eval_llm:
    _type: nim
    model_name: nvidia/nemotron-3-nano-30b-a3b
    temperature: 0.0
```

Reference each by name where needed — tools via `llm_name`, workflow via `llm_name`, evaluators via `llm_name`.
