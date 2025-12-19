# Milestone 1: Agent Spec Import Support in NeMo Agent Toolkit

Author: Oracle (Yasha Pushak) • Reviewer: NVIDIA (NeMo team)

Status: Draft (Implementation Plan)

## Scope

Add a new workflow/function type to NeMo Agent Toolkit that executes Agent Spec configurations using the Agent Spec → LangGraph adapter. This enables representing richer agentic patterns (flows, branches/loops, multi‑agent) inside NeMo, while reusing NeMo’s tooling (evaluation, profiling, observability).

Non-goals for this milestone:
- Multi-runtime selection (CrewAI, WayFlow, …) — planned for Milestone 2.
- Sub-component tuning via Agent Spec graph introspection — planned for Milestone 3.
- New UI or API surface beyond what is required to run an Agent Spec workflow.

## High-level Design

- Introduce a new function config type: `AgentSpecWorkflowConfig` with `_type: agent_spec`.
- At build time, use the Agent Spec LangGraph adapter to convert an Agent Spec document into a `langgraph` CompiledStateGraph and run it as the workflow implementation.
- Reuse NeMo’s Builder to construct a tool registry compatible with LangGraph/LangChain tools from NAT-configured tools.
- Keep inputs/outputs aligned with NeMo’s `ChatRequest`/`ChatResponse` for consistent UX.
- Hook into NeMo profiling/observability using existing `LLMFrameworkEnum.LANGCHAIN` wrapper integration.

## Key Integration Points

- Registration: `@register_function(config_type=AgentSpecWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])`.
- Config model: subclass `AgentBaseConfig` (for parity with `react_agent`), add fields to accept the Agent Spec payload and minimal runtime options.
- Adapter usage: `langgraph_agentspec_adapter.AgentSpecLoader` to `load_yaml`/`load_json` or from a structured Pydantic field converted to YAML/JSON.
- Tools: leverage `builder.get_tools(..., wrapper_type=LLMFrameworkEnum.LANGCHAIN)` to obtain LangChain-compatible tools, then pass them into the adapter `tool_registry` by name.
- Execution: compile LangGraph component from Agent Spec and `ainvoke` based on `ChatRequest` messages; return `ChatResponse` with basic usage stats, consistent with existing agents.

## Configuration Schema (initial)

Example NAT YAML:

```yaml
workflow:
  _type: agent_spec
  description: Agent Spec workflow
  # Exactly one of the following should be provided
  agentspec_yaml: |
    component_type: Agent
    id: bdd2369b-82e6-488f-be2c-44f05b244cab
    name: writing agent
    description: Agent to help write blog articles
    system_prompt: "You're a helpful writing assistant..."
    inputs:
      - title: user_name
        type: string
    llm_config:
      component_type: VllmConfig
      model_id: gpt-oss-120b
      url: http://url.to.my.vllm.server/
    tools:
      - component_type: ServerTool
        name: pretty_formatting
        description: Format paragraph spacing and indentation
        inputs:
          - title: paragraph
            type: string
        outputs:
          - title: formatted_paragraph
            type: string
  # Optional alternatives to inline YAML
  # agentspec_json: "{...}"
  # agentspec_path: path/to/agent_spec.yaml
  tool_names: [pretty_formatting]  # map to NeMo/LC tools where applicable
  verbose: true
  max_history: 15
```

Notes:
- We support exactly one of: `agentspec_yaml`, `agentspec_json`, or `agentspec_path`.
- `tool_names` are optional. If provided, we populate the adapter `tool_registry` with LC-compatible wrappers for those names. If the Agent Spec includes ServerTools with embedded `func`, both can coexist; the registry complements the spec.
- We reuse standard agent flags (`verbose`, `max_history`, `log_response_max_chars`, etc.) when easy; reserved flags beyond minimal needs can be deferred.

## Detailed Tasks

1) Data model and registration
- Add `AgentSpecWorkflowConfig` in `nat/agent/agentspec/register.py`:
  - Base: `AgentBaseConfig`.
  - Name: `agent_spec`.
  - Fields:
    - `description: str = "Agent Spec Workflow"`.
    - `agentspec_yaml: str | None`.
    - `agentspec_json: str | None`.
    - `agentspec_path: str | None`.
    - `tool_names: list[FunctionRef | FunctionGroupRef] = []` (optional).
    - `max_history: int = 15`, `verbose: bool = False`, `log_response_max_chars: int | None = None`.
    - Validation: exactly one of `agentspec_yaml/json/path` must be set.
- Register build function with `@register_function(..., framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])`.

2) Builder integration (execution function)
- Resolve the Agent Spec payload (from YAML/JSON/path) into a string.
- Build LC-compatible tools via `builder.get_tools(tool_names=..., wrapper_type=LLMFrameworkEnum.LANGCHAIN)` and create a dict `{name: tool}` to pass as `tool_registry` to `AgentSpecLoader`.
- Instantiate adapter: `AgentSpecLoader(tool_registry=..., checkpointer=None, config=None)`.
- Load to LangGraph component via `load_yaml`/`load_json`.
- Implement NAT function body accepting `ChatRequestOrMessage`:
  - Convert to `ChatRequest`.
  - Trim/limit history based on `max_history` (reuse logic from `react_agent`).
  - Prepare input state/messages as required by the compiled graph:
    - If the compiled component expects `{"messages": [...]}`, pass aligned payload.
    - Otherwise, pass the minimal input the adapter expects; start with messages-only path as per adapter examples.
  - `await graph.ainvoke(state, config={"recursion_limit": safe_default})`.
  - Extract final message content and build `ChatResponse` with basic token usage.

3) Dependency management
- Runtime dependency on `langgraph-agentspec-adapter` and `pyagentspec`.
- Add an extra in NeMo toolkit packaging (e.g., `agentspec`) and document installation: `pip install nat[agentspec]`.
- For source builds in this mono-repo, ensure import path resolution works during tests (use dev requirements or local path instruction in docs).

4) Error handling and validation
- Clear errors for:
  - Missing/invalid Agent Spec payload.
  - Both/none of YAML/JSON/path provided.
  - Tool name not found in NAT registry when specified.
  - Adapter conversion/type errors.
- Log with existing agent log prefix; respect `verbose` and `log_response_max_chars` where applicable.

5) Observability/profiling
- Mark function with `framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]` to enable existing profiler hooks.
- Ensure intermediate steps respect NAT callback/logging conventions where feasible in this milestone (best effort; deep step-by-step surfacing can be deferred).

6) Documentation
- Add a new page under `docs/source/components/agents/agent-spec.md`:
  - Purpose and capabilities.
  - Installation instructions (extra deps).
  - Minimal YAML example and tool mapping guidance.
  - Notes/limitations in Milestone 1.
- Update tutorials index and examples list.

7) Testing
- Unit tests:
  - Config validation (exactly-one-of source fields).
  - Tool registry mapping (with and without `tool_names`).
- Adapter smoke test:
  - Use a tiny Agent Spec with a single tool or echo flow; verify invocation returns a `ChatResponse`.
  - Mark network/LLM usage as skipped unless a local model is configured; prefer a spec that does not require external LLM for the smoke test (e.g., a simple tool graph), or mock the adapter/LLM call in tests.

## Open Questions / Assumptions

- Tool mapping precedence: if an Agent Spec defines ServerTools and NAT also provides `tool_names`, we will merge, with NAT tools overwriting duplicate names in the adapter registry (document this behavior).
- Input schema: In Milestone 1 we standardize on `ChatRequest` messages as input; richer parameter passing (matching Agent Spec inputs) can be extended later.
- Checkpointing: adapter supports a `Checkpointer`; we defer integration to a later milestone unless straightforward to pass through.
- Streaming: we target non-streaming response first; streaming support can be added post-M1 if needed.

## Risks and Mitigations

- Dependency drift between LangGraph/LangChain versions: pin compatible versions in the `agentspec` extra and CI.
- Mismatch between adapter’s expected state shape and our message wrapper: start with the adapter’s documented pattern using `{"messages": [...]}` and add a thin translator if necessary.
- Tool contract mismatch: validate tool signatures with simple probes and surface clear errors.

## Delivery Checklist

- [x] `AgentSpecWorkflowConfig` added and registered.
- [x] Build function executes adapter graph and returns `ChatResponse`.
- [x] Packaging: `agentspec` extra with pinned dependencies and docs stub.
- [ ] Docs page with examples and limitations.
- [x] Unit tests: config validation.
- [ ] Smoke test with minimal Agent Spec graph.

## Plan Updates

- Added optional dependency extra `agentspec` in `pyproject.toml` with `pyagentspec` and `langgraph-agentspec-adapter`.
- Implemented `AgentSpecWorkflowConfig` at `src/nat/agent/agentspec/register.py:1` and wired registration in `src/nat/agent/register.py:1`.
- Added basic config validation tests in `tests/test_agentspec_config.py:1`.

## Appendix: References

- NAT registration pattern: `nat/agent/react_agent/register.py` (`ReActAgentWorkflowConfig`, `react_agent_workflow`).
- Type registry and wrappers: `nat/cli/register_workflow.py`, `LLMFrameworkEnum.LANGCHAIN`.
- Adapter entrypoints: `langgraph_agentspec_adapter.AgentSpecLoader` (load_yaml/json/component).
