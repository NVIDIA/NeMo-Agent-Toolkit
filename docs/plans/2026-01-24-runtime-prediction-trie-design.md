# Runtime Prediction Trie Integration Design

## Overview

This design addresses the gap between the prediction trie (built by the profiler) and runtime execution. Currently, the trie is built and saved, but never loaded or used during actual workflow execution to inject prediction headers.

## Problem Statement

The prediction trie implementation has the following gaps:

1. **Trie never loaded at runtime** - `prediction_trie_path` config exists but is never used
2. **Function path not tracked for lookups** - `Context.active_function` only stores immediate parent, not full ancestry
3. **Call index never tracked at runtime** - `LLMCallTracker` exists but is never incremented during LLM calls
4. **Headers are static** - httpx client created once with static hooks; predictions need dynamic per-call lookup

## Design Goals

- Track full function path ancestry during workflow execution
- Track LLM call indices per parent function
- Look up predictions dynamically on each LLM call
- Inject prediction headers for Dynamo routing optimization
- Work across all LLM frameworks (LangChain, LlamaIndex, etc.)

## Architecture

### Separation of Concerns

| Concern | Component | Scope |
|---------|-----------|-------|
| State tracking | Callback handlers + IntermediateStepManager | All LLM providers |
| Header injection | Dynamo httpx hook | Dynamo LLM only |

This separation ensures state is tracked universally (even if multiple LLM providers are used in one workflow), while header injection is specific to Dynamo.

### Data Flow

```
1. Workflow starts
   └─► function_path_stack = ["my_workflow"]

2. Agent function called via push_active_function("react_agent")
   └─► function_path_stack = ["my_workflow", "react_agent"]

3. LLM call initiated
   └─► Callback fires on_chat_model_start
       └─► IntermediateStepManager.push_intermediate_step(LLM_START)
           └─► call_tracker.increment(parent_function_id) → 1

4. httpx sends request (Dynamo)
   └─► Dynamic hook executes:
       ├─► Read function_path_stack → ["my_workflow", "react_agent"]
       ├─► Read call_tracker count → 1
       ├─► trie_lookup.find(path, call_index) → prediction
       │   └─► (fallback to root.predictions_any_index if no match)
       └─► Inject headers

5. Next LLM call → call_index becomes 2, repeat
```

## Components to Modify

### 1. ContextState (src/nat/builder/context.py)

Add new ContextVar to track full function path:

```python
class ContextState:
    def __init__(self):
        # ... existing fields ...
        self._function_path_stack: ContextVar[list[str] | None] = ContextVar(
            "function_path_stack", default=None
        )

    @property
    def function_path_stack(self) -> ContextVar[list[str]]:
        if self._function_path_stack.get() is None:
            self._function_path_stack.set([])
        return typing.cast(ContextVar[list[str]], self._function_path_stack)
```

### 2. Context.push_active_function() (src/nat/builder/context.py)

Update to push/pop function names on path stack:

```python
@contextmanager
def push_active_function(self, function_name: str, ...):
    # ... existing code ...

    # Push function name onto path stack
    current_path = self._context_state.function_path_stack.get()
    new_path = current_path + [function_name]
    path_token = self._context_state.function_path_stack.set(new_path)

    try:
        yield manager
    finally:
        # ... existing cleanup ...
        self._context_state.function_path_stack.reset(path_token)
```

### 3. IntermediateStepManager.push_intermediate_step() (src/nat/builder/intermediate_step_manager.py)

Increment call tracker on LLM_START events:

```python
from nat.llm.prediction_context import get_call_tracker

def push_intermediate_step(self, payload: IntermediateStepPayload) -> None:
    # ... existing code ...

    # Track LLM call index for prediction lookups
    if payload.event_type == IntermediateStepType.LLM_START:
        active_function = self._context_state.active_function.get()
        if active_function:
            tracker = get_call_tracker()
            tracker.increment(active_function.function_id)

    # ... rest of existing code ...
```

### 4. Context.function_path property (src/nat/builder/context.py)

Add property to read current function path:

```python
@property
def function_path(self) -> list[str]:
    """Returns the current function path stack (copy)."""
    return list(self._context_state.function_path_stack.get())
```

### 5. dynamo_langchain() (packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py)

Load trie and create dynamic hook:

```python
from nat.profiler.prediction_trie import load_prediction_trie, PredictionTrieLookup

@register_llm_client(config_type=DynamoModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def dynamo_langchain(llm_config: DynamoModelConfig, _builder: Builder):
    # Load prediction trie if configured
    trie_lookup: PredictionTrieLookup | None = None
    if llm_config.prediction_trie_path:
        trie = load_prediction_trie(Path(llm_config.prediction_trie_path))
        trie_lookup = PredictionTrieLookup(trie)
        logger.info("Loaded prediction trie from %s", llm_config.prediction_trie_path)

    # Create httpx client with dynamic prediction hook
    if llm_config.prefix_template is not None:
        http_async_client = create_httpx_client_with_dynamo_hooks(
            # ... existing params ...
            prediction_lookup=trie_lookup,  # Pass lookup to hook
        )
```

### 6. Dynamic Prediction Hook (src/nat/llm/dynamo_llm.py)

Create hook that reads context and looks up predictions:

```python
def _create_dynamic_prediction_hook(
    trie_lookup: PredictionTrieLookup,
) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """Create hook that dynamically looks up predictions per request."""

    async def on_request(request: "httpx.Request") -> None:
        from nat.builder.context import Context
        from nat.llm.prediction_context import get_call_tracker

        ctx = Context.get()
        path = ctx.function_path

        # Get call index for current parent function
        call_index = 1  # default
        active_fn = ctx.active_function
        if active_fn:
            tracker = get_call_tracker()
            call_index = tracker.counts.get(active_fn.function_id, 1)

        # Look up prediction
        prediction = trie_lookup.find(path, call_index)

        if prediction:
            request.headers["x-nat-remaining-llm-calls"] = str(int(prediction.remaining_calls.mean))
            request.headers["x-nat-interarrival-ms"] = str(int(prediction.interarrival_ms.mean))
            request.headers["x-nat-expected-output-tokens"] = str(int(prediction.output_tokens.p90))

            logger.debug(
                "Injected prediction headers: path=%s, call_index=%d, remaining=%d",
                path, call_index, int(prediction.remaining_calls.mean)
            )

    return on_request
```

## Fallback Chain

When looking up predictions, the following fallback chain applies:

1. **Exact match**: path + call_index found in trie
2. **Partial path**: walk trie as far as possible, use deepest match
3. **Any index**: use node's `predictions_any_index` if exact call_index not found
4. **Root fallback**: use root's `predictions_any_index` as final fallback

This ensures we always have some prediction to inject (root aggregates across all profiled traces).

## Call Index Tracking

- Each function invocation has a unique UUID (`function_id`)
- `LLMCallTracker.increment(function_id)` returns 1, 2, 3... for successive LLM calls
- No explicit reset needed - new function invocations get new UUIDs automatically
- Memory is minimal (dict of int counters) and garbage collected with context

## Headers Injected

| Header | Value | Description |
|--------|-------|-------------|
| `x-nat-remaining-llm-calls` | `int(prediction.remaining_calls.mean)` | Expected remaining LLM calls |
| `x-nat-interarrival-ms` | `int(prediction.interarrival_ms.mean)` | Expected ms until next call |
| `x-nat-expected-output-tokens` | `int(prediction.output_tokens.p90)` | Expected output tokens (p90) |

## Testing Strategy

1. **Unit tests**: Test each component in isolation
   - `function_path_stack` push/pop behavior
   - Call tracker increment in IntermediateStepManager
   - Dynamic hook reads context correctly

2. **Integration test**: End-to-end flow
   - Create trie from sample traces
   - Run workflow with Dynamo LLM
   - Verify headers injected with correct values

3. **Fallback test**: Verify fallback chain
   - Unknown path falls back to root
   - Unknown call_index falls back to any_index

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `src/nat/builder/context.py` | Modify | Add function_path_stack ContextVar and property |
| `src/nat/builder/intermediate_step_manager.py` | Modify | Increment call tracker on LLM_START |
| `src/nat/llm/dynamo_llm.py` | Modify | Add dynamic prediction hook |
| `packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py` | Modify | Load trie, wire up hook |
| `tests/nat/builder/test_function_path_stack.py` | New | Test path stack tracking |
| `tests/nat/llm/test_dynamic_prediction_hook.py` | New | Test dynamic lookup and injection |
