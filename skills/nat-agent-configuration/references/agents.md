# Built-in Agent Types

The nine built-in NeMo Agent toolkit agent types: when to pick which, key traits, and YAML examples. Companion: [`additional-agent-types.md`](additional-agent-types.md) lists the lesser-used types in detail.

**Classify the workflow shape before scaffolding:**

- Fixed steps with no LLM decision-making → `sequential_executor` / `parallel_executor` or a plain function.
- Research, analyze, decide, summarize, RAG, MCP, iterative tool use, "build an agent that…" → an agent type below.
- **If ambiguous, ask one clarifying question before writing YAML.**

| Agent | `_type` | LLM Required | Uses Tools | Key Trait |
| --- | --- | --- | --- | --- |
| ReAct | `react_agent` | Yes | Yes | Interleaved reason + act loop |
| Reasoning | `reasoning_agent` | Yes (thinking-capable) | Via inner agent | Plan-then-delegate wrapper |
| ReWOO | `rewoo_agent` | Yes | Yes | Separate plan / execute / solve phases |
| Responses API | `responses_api_agent` | Yes (Responses API) | NeMo Agent toolkit + built-in + MCP | Direct LLM tool binding with MCP support |
| Router | `router_agent` | Yes | No (routes to branches) | LLM-based single-branch dispatch |
| Tool Calling | `tool_calling_agent` | Yes | Yes | Standard tool-calling API, direct output |
| Auto Memory Wrapper | `auto_memory_agent` | Yes | Via inner agent | Persistent memory across turns |
| Parallel Executor | `parallel_executor` | No | Yes (fixed list) | Concurrent fan-out/fan-in pipeline |
| Sequential Executor | `sequential_executor` | No | Yes (fixed list) | Deterministic linear pipeline |

## ReAct Agent (`react_agent`)

Alternates between reasoning and acting: think -> call tool -> observe result -> repeat until done (up to `max_tool_calls`, default 15).

**Use when:** Tasks require interleaved reasoning over intermediate tool results; the LLM does not reliably support native tool calling (ReAct falls back to text-based `Action:`/`Action Input:` parsing); tools lack clean JSON schemas.

**Avoid when:** Tools have well-defined schemas (prefer `tool_calling_agent`); parallel tool execution is needed; response format must be guaranteed; the task is a simple single-tool lookup; the model is a reasoning model that uses thinking tokens.

```yaml
workflow:
  _type: react_agent
  tool_names: [wikipedia_search, current_datetime]
  llm_name: nim_llm
  verbose: true
```

**Native tool calling vs text-based parsing:** The ReAct agent supports two modes for tool invocation:

- `use_native_tool_calling: false` (default) — text-based ReAct parsing. The agent generates text with `Action:` and `Action Input:` markers. **More reliable across different LLM providers.** Use this as the default.
- `use_native_tool_calling: true` — uses the LLM's function/tool-calling API. More structured but **requires the LLM endpoint to support OpenAI-compatible tool calling.** Some providers (including certain NVIDIA Inference API models) may not invoke tools correctly in this mode. If the agent generates responses without calling any tools, switch to `false`.

```yaml
workflow:
  _type: react_agent
  tool_names: [wikipedia_search, current_datetime]
  llm_name: nim_llm
  verbose: true
  # Start with false (text-based). Switch to true only if the LLM
  # reliably supports native tool calling.
  use_native_tool_calling: false
```

**Do not override the agent's tool-calling format in the workflow input.** The ReAct agent has a built-in system prompt that instructs the LLM on the correct `Action:` / `Action Input:` format. Adding format instructions to the workflow input (the user query) will not help and may conflict with the system prompt. If the LLM still outputs the wrong format (e.g., `<function_calls>` XML tags), the fix is one of:

1. Try a different LLM that follows the ReAct format reliably
2. Switch `use_native_tool_calling: true` if the endpoint supports OpenAI-compatible tool calling
3. Use a different agent type (`tool_calling_agent` for simpler cases, `router_agent` for single-branch dispatch)

Keep the workflow input focused on the task description only.

**Temperature for ReAct agents:** Use `temperature: 0.0` for deterministic, reproducible tool-calling behavior. Higher temperatures (0.3–0.7) can improve output diversity in the final synthesis step but may cause the agent to skip tool calls or hallucinate action formats. When optimizing agent output quality, adjust the prompt instructions before raising temperature.

**Hint:** Can be used when a pure LLM workflow would be sufficient to solve a task. Even though in that case no tool would be necessary, the ReAct agent requires at least one tool. Just add the built-in `current_datetime` tool so the list is not empty.

## Tool Calling Agent (`tool_calling_agent`)

Uses the LLM's native function/tool-calling API to select and invoke tools. Returns the tool's structured output directly, without the ReAct reasoning loop. Can iterate up to `max_iterations` (default 15).

**Use when:** Tools have well-defined JSON schemas — especially **MCP tools** (via `mcp_client`) and function groups with `input_schema`. You want structured output returned directly. The LLM endpoint supports OpenAI-compatible tool calling. This is the preferred choice for MCP-based agents.

**Avoid when:** The LLM doesn't support native tool calling reliably (use `react_agent` with text parsing instead); tasks require interleaved reasoning over intermediate results before deciding the next action.

```yaml
workflow:
  _type: tool_calling_agent
  tool_names: [kb_tools]
  llm_name: nvidia_llm
  verbose: true
  handle_tool_errors: true   # catch tool exceptions and pass them back to the LLM for recovery
```

## Sequential Executor (`sequential_executor`)

Deterministic, LLM-free pipeline. Executes a fixed list of functions in order, passing each function's output as the next function's input. No reasoning involved. **Used by** the Demographics Repartition agent for chained generation workflows.

**Use when:** Processing data through multiple sequential transformation stages with predictable, hard-coded execution order.

**Avoid when:** Conditional branching, parallel processing, or any form of LLM-guided decision-making is needed.

```yaml
workflow:
  _type: sequential_executor
  tool_list: [text_processor, data_analyzer, report_generator]
```

## Running Agents Programmatically

When you need a Python entry point that invokes a NeMo Agent toolkit workflow, use the programmatic API instead of shelling out:

```python
import asyncio
from pathlib import Path

from nat.builder.workflow_builder import WorkflowBuilder
from nat.runtime.loader import load_config


async def run_workflow(query: str) -> str:
    # 1. Load YAML config
    config = load_config("workflow.yaml")

    # 2. Build the workflow (registers all components)
    async with WorkflowBuilder.from_config(config) as builder:
        workflow = await builder.build()

        # 3. Run the workflow and get the result
        async with workflow.run(query) as runner:
            result = await runner.result()

    return str(result)


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Hello"
    print(asyncio.run(run_workflow(query)))
```

**Key points:**

- `WorkflowBuilder.from_config()` is an async context manager — always use `async with`
- `builder.build()` returns a `Workflow` object
- `workflow.run(query)` is also an async context manager — use `async with` to get a `runner`
- Call `await runner.result()` to get the final answer as a string
