# Additional NeMo Agent Toolkit Agent Types

These agent types are available in NeMo Agent Toolkit but are less frequently used in practice compared to ReAct and Sequential Executor.

## Reasoning Agent (`reasoning_agent`)

A wrapper that first uses a thinking-capable LLM to generate a detailed plan, then passes that plan + original query to an inner agent (`augmented_fn`) for execution.

**Use when:** Complex problems that benefit from structured upfront thinking before execution; wrapping an existing agent to improve its performance.

**Avoid when:** Latency is critical (plan generation adds overhead) or the LLM doesn't support thinking/reasoning.

```yaml
workflow:
  _type: reasoning_agent
  llm_name: nemotron_model
  # The augmented_fn is the nat Function that the execution plan is passed to. Usually an agent entry point.
  augmented_fn: react_agent
  verbose: true
```

To use a custom inner agent with its own tools and LLM, define it under `functions:` and reference it by key:

```yaml
functions:
  inner_agent:
    _type: react_agent
    tool_names: [current_datetime]
    llm_name: react_llm

workflow:
  _type: reasoning_agent
  llm_name: reasoning_llm
  augmented_fn: inner_agent
  verbose: true
```

## ReWOO Agent (`rewoo_agent`)

Three-phase: (1) planner LLM decomposes the task into a full ordered plan with placeholders, (2) all tools execute sequentially filling the placeholders, (3) solver LLM synthesizes the final answer.

**Use when:** Complex tasks needing upfront decomposition with clear planning/execution separation.

**Avoid when:** Real-time applications (upfront planning adds latency) or tools need parallel execution.

```yaml
workflow:
  _type: rewoo_agent
  tool_names: [wikipedia_search, current_datetime]
  llm_name: nim_llm
  verbose: true
  use_tool_schema: true   # pass each tool's JSON input schema to the planner so it generates valid arguments
```

## Responses API Agent (`responses_api_agent`)

Uses the OpenAI Responses API to bind tools directly to the LLM. Supports three tool categories simultaneously: NeMo Agent Toolkit tools, built-in LLM tools (e.g. code interpreter), and remote MCP tools.

**Use when:** The model supports the Responses API; you need built-in tools like code execution or remote MCP integrations.

**Avoid when:** The model doesn't support the Responses API, or simpler patterns (ReAct, ReWOO) suffice.

```yaml
functions:
  current_datetime:
    _type: current_datetime

llms:
  openai_llm:
    _type: openai
    model_name: gpt-4o-mini
    api_type: responses

workflow:
  _type: responses_api_agent
  llm_name: openai_llm
  nat_tools: [current_datetime]
  builtin_tools:
    - type: code_interpreter
      container:
        type: auto
```

## Router Agent (`router_agent`)

Two-phase: the LLM classifies the request and routes it to exactly one branch (tool, function, or sub-agent) from a predefined list.

**Use when:** Requests need to be directed to multiple specialized agents or tools; multi-domain systems with distinct handlers per domain.

**Avoid when:** Sequential execution across multiple branches is needed, or branches have interdependent outputs (only one branch is invoked).

```yaml
workflow:
  _type: router_agent
  branches: [fruit_advisor, city_advisor, literature_advisor]
  llm_name: nim_llm
```

## Parallel Executor (`parallel_executor`)

Fans out the same input to all listed tools concurrently, waits for all to complete, then merges their outputs into a single response. No LLM involved — purely deterministic concurrent execution.

**Use when:** Multiple independent tools need to process the same input simultaneously and their outputs should be combined (e.g., querying multiple data sources, running parallel analyses).

**Avoid when:** Tools depend on each other's output (use Sequential Executor), or the decision of which tool to call needs LLM reasoning (use Router or ReAct).

```yaml
functions:
  analyze_sentiment:
    _type: analyze_sentiment
  extract_entities:
    _type: extract_entities
  summarize_text:
    _type: summarize_text

workflow:
  _type: parallel_executor
  tool_list: [analyze_sentiment, extract_entities, summarize_text]
  description: "Run sentiment, entity, and summary analysis in parallel"
```

Optional settings:

- `detailed_logs: true` — logs fan-out, per-branch timing, and fan-in events
- `return_error_on_exception: true` — captures branch errors as output instead of raising

## Auto Memory Wrapper (`auto_memory_agent`)

Wraps any inner agent to add persistent memory: retrieves relevant history before each call, stores user messages and responses in a memory backend after each call.

**Use when:** Multi-turn dialogue systems needing context from previous sessions.

**Avoid when:** Single-turn stateless interactions, privacy constraints prevent history retention, or memory backend latency is unacceptable.

```yaml
memory:
  zep_memory:
    _type: nat.plugins.zep_cloud/zep_memory

functions:
  my_react_agent:
    _type: react_agent
    llm_name: nim_llm
    tool_names: [calculator]

workflow:
  _type: auto_memory_agent
  inner_agent_name: my_react_agent
  memory_name: zep_memory
  llm_name: nim_llm
```
