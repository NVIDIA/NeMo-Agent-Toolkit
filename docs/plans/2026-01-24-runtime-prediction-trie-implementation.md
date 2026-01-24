# Runtime Prediction Trie Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable runtime prediction trie lookups to inject Dynamo headers based on current function path and LLM call index.

**Architecture:** Add a function path stack ContextVar for tracking ancestry, increment call tracker in IntermediateStepManager on LLM_START events, and create a dynamic httpx hook that reads context and looks up predictions from a pre-loaded trie.

**Tech Stack:** Python 3.11+, contextvars, Pydantic v2, httpx event hooks

---

## Task 1: Add Function Path Stack to ContextState

**Files:**
- Modify: `src/nat/builder/context.py:67-120`
- Test: `tests/nat/builder/test_function_path_stack.py`

### Step 1: Write the failing test for function_path_stack

```python
# tests/nat/builder/test_function_path_stack.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.builder.context import ContextState


def test_function_path_stack_default_empty():
    """Test that function_path_stack starts empty."""
    state = ContextState.get()
    # Reset to test fresh state
    state._function_path_stack.set(None)

    path = state.function_path_stack.get()
    assert path == []


def test_function_path_stack_can_be_set():
    """Test that function_path_stack can be set and retrieved."""
    state = ContextState.get()
    state.function_path_stack.set(["workflow", "agent"])

    path = state.function_path_stack.get()
    assert path == ["workflow", "agent"]
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/builder/test_function_path_stack.py::test_function_path_stack_default_empty -v`
Expected: FAIL with "AttributeError: 'ContextState' object has no attribute '_function_path_stack'"

### Step 3: Add function_path_stack ContextVar to ContextState

In `src/nat/builder/context.py`, add to `ContextState.__init__` after line 83:

```python
        self._function_path_stack: ContextVar[list[str] | None] = ContextVar("function_path_stack", default=None)
```

And add the property after `active_span_id_stack` property (after line 116):

```python
    @property
    def function_path_stack(self) -> ContextVar[list[str]]:
        if self._function_path_stack.get() is None:
            self._function_path_stack.set([])
        return typing.cast(ContextVar[list[str]], self._function_path_stack)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/nat/builder/test_function_path_stack.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/builder/context.py tests/nat/builder/test_function_path_stack.py
git commit --signoff -m "feat(context): add function_path_stack ContextVar

Tracks the full function ancestry path as a list of function names,
enabling prediction trie lookups at runtime."
```

---

## Task 2: Update push_active_function to Track Path Stack

**Files:**
- Modify: `src/nat/builder/context.py:235-279`
- Test: `tests/nat/builder/test_function_path_stack.py`

### Step 1: Write the failing test for push_active_function path tracking

Add to `tests/nat/builder/test_function_path_stack.py`:

```python
from nat.builder.context import Context


def test_push_active_function_updates_path_stack():
    """Test that push_active_function pushes/pops from path stack."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset path stack
    state._function_path_stack.set(None)

    # Initially empty
    assert state.function_path_stack.get() == []

    with ctx.push_active_function("my_workflow", input_data=None):
        assert state.function_path_stack.get() == ["my_workflow"]

        with ctx.push_active_function("react_agent", input_data=None):
            assert state.function_path_stack.get() == ["my_workflow", "react_agent"]

            with ctx.push_active_function("tool_call", input_data=None):
                assert state.function_path_stack.get() == ["my_workflow", "react_agent", "tool_call"]

            # After tool_call exits
            assert state.function_path_stack.get() == ["my_workflow", "react_agent"]

        # After react_agent exits
        assert state.function_path_stack.get() == ["my_workflow"]

    # After workflow exits
    assert state.function_path_stack.get() == []
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/builder/test_function_path_stack.py::test_push_active_function_updates_path_stack -v`
Expected: FAIL with assertion error (path stack not being updated)

### Step 3: Update push_active_function to track path stack

In `src/nat/builder/context.py`, modify `push_active_function` method. After line 252 (after setting fn_token), add:

```python
        # 1b) Push function name onto path stack
        current_path = self._context_state.function_path_stack.get()
        new_path = current_path + [function_name]
        path_token = self._context_state.function_path_stack.set(new_path)
```

And in the finally block, before line 279 (before resetting fn_token), add:

```python
            # 4a) Pop function name from path stack
            self._context_state.function_path_stack.reset(path_token)
```

### Step 4: Run test to verify it passes

Run: `pytest tests/nat/builder/test_function_path_stack.py -v`
Expected: PASS

### Step 5: Add function_path property to Context class

Add after `active_function` property (around line 289):

```python
    @property
    def function_path(self) -> list[str]:
        """
        Returns a copy of the current function path stack.

        The function path represents the ancestry of the currently executing
        function, from root to the current function.

        Returns:
            list[str]: Copy of the function path stack.
        """
        return list(self._context_state.function_path_stack.get())
```

### Step 6: Write test for function_path property

Add to `tests/nat/builder/test_function_path_stack.py`:

```python
def test_context_function_path_property():
    """Test that Context.function_path returns a copy of the path stack."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset path stack
    state._function_path_stack.set(None)

    with ctx.push_active_function("workflow", input_data=None):
        with ctx.push_active_function("agent", input_data=None):
            path = ctx.function_path
            assert path == ["workflow", "agent"]

            # Verify it's a copy (modifications don't affect original)
            path.append("modified")
            assert ctx.function_path == ["workflow", "agent"]
```

### Step 7: Run all tests

Run: `pytest tests/nat/builder/test_function_path_stack.py -v`
Expected: PASS

### Step 8: Commit

```bash
git add src/nat/builder/context.py tests/nat/builder/test_function_path_stack.py
git commit --signoff -m "feat(context): track function path in push_active_function

Push/pop function names onto function_path_stack in push_active_function.
Add Context.function_path property to retrieve the current path."
```

---

## Task 3: Increment Call Tracker in IntermediateStepManager

**Files:**
- Modify: `src/nat/builder/intermediate_step_manager.py:64-96`
- Test: `tests/nat/builder/test_call_tracker_integration.py`

### Step 1: Write the failing test for call tracker integration

```python
# tests/nat/builder/test_call_tracker_integration.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.llm.prediction_context import get_call_tracker


def test_llm_start_increments_call_tracker():
    """Test that pushing an LLM_START step increments the call tracker."""
    ctx = Context.get()
    step_manager = ctx.intermediate_step_manager

    with ctx.push_active_function("test_agent", input_data=None):
        active_fn = ctx.active_function
        tracker = get_call_tracker()

        # Initially no count for this function
        assert tracker.counts.get(active_fn.function_id, 0) == 0

        # Push LLM_START
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="llm-call-1",
                event_type=IntermediateStepType.LLM_START,
                name="test-model",
            )
        )

        # Call tracker should be incremented
        assert tracker.counts.get(active_fn.function_id) == 1

        # Push another LLM_START
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="llm-call-2",
                event_type=IntermediateStepType.LLM_START,
                name="test-model",
            )
        )

        # Should be 2 now
        assert tracker.counts.get(active_fn.function_id) == 2


def test_non_llm_start_does_not_increment_tracker():
    """Test that non-LLM_START events don't increment the tracker."""
    ctx = Context.get()
    step_manager = ctx.intermediate_step_manager

    with ctx.push_active_function("test_agent_2", input_data=None):
        active_fn = ctx.active_function
        tracker = get_call_tracker()

        initial_count = tracker.counts.get(active_fn.function_id, 0)

        # Push TOOL_START (should not increment)
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="tool-call-1",
                event_type=IntermediateStepType.TOOL_START,
                name="test-tool",
            )
        )

        # Count should be unchanged
        assert tracker.counts.get(active_fn.function_id, 0) == initial_count
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/builder/test_call_tracker_integration.py::test_llm_start_increments_call_tracker -v`
Expected: FAIL with assertion error (count is 0, not 1)

### Step 3: Add call tracker increment to IntermediateStepManager

In `src/nat/builder/intermediate_step_manager.py`, add import at top:

```python
from nat.data_models.intermediate_step import IntermediateStepType
from nat.llm.prediction_context import get_call_tracker
```

Then in `push_intermediate_step` method, after line 96 (after the debug log for START), add:

```python
            # Track LLM call index for prediction trie lookups
            if payload.event_type == IntermediateStepType.LLM_START:
                active_function = self._context_state.active_function.get()
                if active_function and active_function.function_id != "root":
                    tracker = get_call_tracker()
                    tracker.increment(active_function.function_id)
                    logger.debug("Incremented LLM call tracker for %s to %d",
                                 active_function.function_id,
                                 tracker.counts.get(active_function.function_id, 0))
```

### Step 4: Run test to verify it passes

Run: `pytest tests/nat/builder/test_call_tracker_integration.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/builder/intermediate_step_manager.py tests/nat/builder/test_call_tracker_integration.py
git commit --signoff -m "feat(step-manager): increment call tracker on LLM_START

IntermediateStepManager now increments LLMCallTracker when an LLM_START
event is pushed. This enables accurate call index tracking for prediction
trie lookups across all LLM frameworks."
```

---

## Task 4: Create Dynamic Prediction Hook

**Files:**
- Modify: `src/nat/llm/dynamo_llm.py`
- Test: `tests/nat/llm/test_dynamic_prediction_hook.py`

### Step 1: Write the failing test for dynamic prediction hook

```python
# tests/nat/llm/test_dynamic_prediction_hook.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nat.builder.context import Context
from nat.llm.dynamo_llm import _create_dynamic_prediction_hook
from nat.llm.prediction_context import get_call_tracker
from nat.profiler.prediction_trie import PredictionTrieLookup
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


@pytest.fixture(name="sample_trie_lookup")
def fixture_sample_trie_lookup() -> PredictionTrieLookup:
    """Create a sample trie lookup for testing."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    agent_node = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={1: prediction, 2: prediction},
        predictions_any_index=prediction,
    )

    workflow_node = PredictionTrieNode(
        name="my_workflow",
        children={"react_agent": agent_node},
        predictions_any_index=prediction,
    )

    root = PredictionTrieNode(
        name="root",
        children={"my_workflow": workflow_node},
        predictions_any_index=prediction,
    )

    return PredictionTrieLookup(root)


class MockRequest:
    """Mock httpx.Request for testing."""

    def __init__(self):
        self.headers = {}


async def test_dynamic_hook_injects_headers(sample_trie_lookup):
    """Test that dynamic hook injects prediction headers based on context."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset state
    state._function_path_stack.set(None)

    hook = _create_dynamic_prediction_hook(sample_trie_lookup)

    with ctx.push_active_function("my_workflow", input_data=None):
        with ctx.push_active_function("react_agent", input_data=None):
            # Simulate LLM call tracker increment (normally done by step manager)
            tracker = get_call_tracker()
            tracker.increment(ctx.active_function.function_id)

            request = MockRequest()
            await hook(request)

            assert "x-nat-remaining-llm-calls" in request.headers
            assert request.headers["x-nat-remaining-llm-calls"] == "3"
            assert request.headers["x-nat-interarrival-ms"] == "500"
            assert request.headers["x-nat-expected-output-tokens"] == "200"


async def test_dynamic_hook_uses_root_fallback(sample_trie_lookup):
    """Test that hook falls back to root prediction for unknown paths."""
    ctx = Context.get()
    state = ctx._context_state

    # Reset state
    state._function_path_stack.set(None)

    hook = _create_dynamic_prediction_hook(sample_trie_lookup)

    with ctx.push_active_function("unknown_workflow", input_data=None):
        tracker = get_call_tracker()
        tracker.increment(ctx.active_function.function_id)

        request = MockRequest()
        await hook(request)

        # Should still inject headers from root fallback
        assert "x-nat-remaining-llm-calls" in request.headers
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/llm/test_dynamic_prediction_hook.py::test_dynamic_hook_injects_headers -v`
Expected: FAIL with "cannot import name '_create_dynamic_prediction_hook'"

### Step 3: Implement dynamic prediction hook

Add to `src/nat/llm/dynamo_llm.py` after the existing `_create_prediction_request_hook` function (around line 383):

```python
def _create_dynamic_prediction_hook(
    trie_lookup: "PredictionTrieLookup",
) -> Callable[["httpx.Request"], Coroutine[Any, Any, None]]:
    """
    Create an httpx event hook that dynamically looks up predictions per request.

    This hook reads the current function path and call index from context,
    looks up the prediction in the trie, and injects headers.

    Args:
        trie_lookup: The PredictionTrieLookup instance to query

    Returns:
        An async function suitable for use as an httpx event hook.
    """
    # Import here to avoid circular imports
    from nat.profiler.prediction_trie import PredictionTrieLookup

    async def on_request(request: "httpx.Request") -> None:
        """Look up prediction from context and inject headers."""
        from nat.builder.context import Context
        from nat.llm.prediction_context import get_call_tracker

        try:
            ctx = Context.get()
            path = ctx.function_path

            # Get call index for current parent function
            call_index = 1  # default
            active_fn = ctx.active_function
            if active_fn and active_fn.function_id != "root":
                tracker = get_call_tracker()
                call_index = tracker.counts.get(active_fn.function_id, 1)

            # Look up prediction
            prediction = trie_lookup.find(path, call_index)

            if prediction:
                request.headers["x-nat-remaining-llm-calls"] = str(int(prediction.remaining_calls.mean))
                request.headers["x-nat-interarrival-ms"] = str(int(prediction.interarrival_ms.mean))
                request.headers["x-nat-expected-output-tokens"] = str(int(prediction.output_tokens.p90))

                logger.debug(
                    "Injected prediction headers: path=%s, call_index=%d, remaining=%d, interarrival=%d, output=%d",
                    path,
                    call_index,
                    int(prediction.remaining_calls.mean),
                    int(prediction.interarrival_ms.mean),
                    int(prediction.output_tokens.p90),
                )
            else:
                logger.debug("No prediction found for path=%s, call_index=%d", path, call_index)

        except Exception as e:
            # Don't fail the request if prediction lookup fails
            logger.warning("Failed to inject prediction headers: %s", e)

    return on_request
```

Also add the import at top of file (after existing TYPE_CHECKING imports):

```python
if TYPE_CHECKING:
    import httpx
    from nat.profiler.prediction_trie import PredictionTrieLookup
```

### Step 4: Run test to verify it passes

Run: `pytest tests/nat/llm/test_dynamic_prediction_hook.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/llm/dynamo_llm.py tests/nat/llm/test_dynamic_prediction_hook.py
git commit --signoff -m "feat(dynamo): add dynamic prediction hook

Creates httpx hook that reads function path and call index from context,
looks up prediction in trie, and injects headers per-request."
```

---

## Task 5: Update create_httpx_client_with_dynamo_hooks

**Files:**
- Modify: `src/nat/llm/dynamo_llm.py:325-355`
- Test: `tests/nat/llm/test_dynamo_prediction_hook.py`

### Step 1: Write test for updated client creation

Add to `tests/nat/llm/test_dynamic_prediction_hook.py`:

```python
from nat.llm.dynamo_llm import create_httpx_client_with_dynamo_hooks


async def test_client_includes_prediction_hook_when_lookup_provided(sample_trie_lookup):
    """Test that client includes prediction hook when trie_lookup is provided."""
    client = create_httpx_client_with_dynamo_hooks(
        prefix_template="test-{uuid}",
        total_requests=10,
        osl="MEDIUM",
        iat="LOW",
        prediction_lookup=sample_trie_lookup,
    )

    # Should have 2 hooks: dynamo prefix + prediction
    assert len(client.event_hooks["request"]) == 2

    await client.aclose()


async def test_client_works_without_prediction_lookup():
    """Test that client works when prediction_lookup is None."""
    client = create_httpx_client_with_dynamo_hooks(
        prefix_template="test-{uuid}",
        total_requests=10,
        osl="MEDIUM",
        iat="LOW",
        prediction_lookup=None,
    )

    # Should have 1 hook: dynamo prefix only
    assert len(client.event_hooks["request"]) == 1

    await client.aclose()
```

### Step 2: Run test to verify it fails

Run: `pytest tests/nat/llm/test_dynamic_prediction_hook.py::test_client_includes_prediction_hook_when_lookup_provided -v`
Expected: FAIL with "unexpected keyword argument 'prediction_lookup'"

### Step 3: Update create_httpx_client_with_dynamo_hooks

Modify `create_httpx_client_with_dynamo_hooks` in `src/nat/llm/dynamo_llm.py`:

```python
def create_httpx_client_with_dynamo_hooks(
    prefix_template: str | None,
    total_requests: int,
    osl: str,
    iat: str,
    timeout: float = 600.0,
    prediction_lookup: "PredictionTrieLookup | None" = None,
) -> "httpx.AsyncClient":
    """
    Create an httpx.AsyncClient with Dynamo prefix header injection.

    This client can be passed to the OpenAI SDK to inject headers at the HTTP level,
    making it framework-agnostic.

    Args:
        prefix_template: Template string with {uuid} placeholder
        total_requests: Expected number of requests for this prefix
        osl: Output sequence length hint (LOW/MEDIUM/HIGH)
        iat: Inter-arrival time hint (LOW/MEDIUM/HIGH)
        timeout: HTTP request timeout in seconds
        prediction_lookup: Optional PredictionTrieLookup for dynamic header injection

    Returns:
        An httpx.AsyncClient configured with Dynamo header injection.
    """
    import httpx

    hooks: list[Callable] = []

    # Add Dynamo prefix hook
    prefix_hook = _create_dynamo_request_hook(prefix_template, total_requests, osl, iat)
    hooks.append(prefix_hook)

    # Add dynamic prediction hook if lookup provided
    if prediction_lookup is not None:
        prediction_hook = _create_dynamic_prediction_hook(prediction_lookup)
        hooks.append(prediction_hook)

    return httpx.AsyncClient(
        event_hooks={"request": hooks},
        timeout=httpx.Timeout(timeout),
    )
```

### Step 4: Run tests to verify they pass

Run: `pytest tests/nat/llm/test_dynamic_prediction_hook.py -v`
Expected: PASS

### Step 5: Commit

```bash
git add src/nat/llm/dynamo_llm.py tests/nat/llm/test_dynamic_prediction_hook.py
git commit --signoff -m "feat(dynamo): add prediction_lookup param to client creation

create_httpx_client_with_dynamo_hooks now accepts optional prediction_lookup
parameter. When provided, adds dynamic prediction hook to inject headers."
```

---

## Task 6: Load Trie in LangChain Dynamo Client

**Files:**
- Modify: `packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py:202-252`
- Test: `tests/nat/plugins/langchain/test_dynamo_trie_loading.py`

### Step 1: Write test for trie loading

```python
# tests/nat/plugins/langchain/test_dynamo_trie_loading.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest

from nat.llm.dynamo_llm import DynamoModelConfig
from nat.profiler.prediction_trie import save_prediction_trie
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


@pytest.fixture(name="trie_file")
def fixture_trie_file():
    """Create a temporary trie file."""
    prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=3.0, p50=3.0, p90=4.0, p95=5.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    root = PredictionTrieNode(
        name="root",
        predictions_by_call_index={1: prediction},
        predictions_any_index=prediction,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "prediction_trie.json"
        save_prediction_trie(root, path, workflow_name="test")
        yield str(path)


def test_dynamo_config_with_valid_trie_path(trie_file):
    """Test that DynamoModelConfig can be created with valid trie path."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path=trie_file,
    )

    assert config.prediction_trie_path == trie_file


def test_dynamo_config_with_nonexistent_trie_path():
    """Test that DynamoModelConfig accepts nonexistent path (validated at load time)."""
    config = DynamoModelConfig(
        base_url="http://localhost:8000/v1",
        model_name="test-model",
        api_key="test-key",
        prediction_trie_path="/nonexistent/path/trie.json",
    )

    # Config creation should succeed; error happens at runtime
    assert config.prediction_trie_path == "/nonexistent/path/trie.json"
```

### Step 2: Run tests

Run: `pytest tests/nat/plugins/langchain/test_dynamo_trie_loading.py -v`
Expected: PASS (config validation already exists)

### Step 3: Update dynamo_langchain to load trie

Modify `packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py`. Add import at top:

```python
from pathlib import Path

from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.prediction_trie import PredictionTrieLookup
```

Then modify the `dynamo_langchain` function (around line 202-252):

```python
@register_llm_client(config_type=DynamoModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def dynamo_langchain(llm_config: DynamoModelConfig, _builder: Builder):
    """
    Create a LangChain ChatOpenAI client for Dynamo with automatic prefix header injection.

    This client injects Dynamo prefix headers at the HTTP transport level using httpx event hooks,
    enabling KV cache optimization and request routing.
    """
    from langchain_openai import ChatOpenAI

    # Build config dict excluding Dynamo-specific and NAT-specific fields
    config_dict = llm_config.model_dump(
        exclude={"type", "thinking", "api_type", *DynamoModelConfig.get_dynamo_field_names()},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )

    # Initialize http_async_client to None for proper cleanup
    http_async_client = None

    # Load prediction trie if configured
    prediction_lookup: PredictionTrieLookup | None = None
    if llm_config.prediction_trie_path:
        try:
            trie_path = Path(llm_config.prediction_trie_path)
            trie = load_prediction_trie(trie_path)
            prediction_lookup = PredictionTrieLookup(trie)
            logger.info("Loaded prediction trie from %s", llm_config.prediction_trie_path)
        except FileNotFoundError:
            logger.warning("Prediction trie file not found: %s", llm_config.prediction_trie_path)
        except Exception as e:
            logger.warning("Failed to load prediction trie: %s", e)

    try:
        # If prefix_template is set, create a custom httpx client with Dynamo hooks
        if llm_config.prefix_template is not None:
            http_async_client = create_httpx_client_with_dynamo_hooks(
                prefix_template=llm_config.prefix_template,
                total_requests=llm_config.prefix_total_requests,
                osl=llm_config.prefix_osl,
                iat=llm_config.prefix_iat,
                timeout=llm_config.request_timeout,
                prediction_lookup=prediction_lookup,
            )
            config_dict["http_async_client"] = http_async_client
            logger.info(
                "Dynamo prefix headers enabled: template=%s, total_requests=%d, osl=%s, iat=%s, prediction_trie=%s",
                llm_config.prefix_template,
                llm_config.prefix_total_requests,
                llm_config.prefix_osl,
                llm_config.prefix_iat,
                "loaded" if prediction_lookup else "disabled",
            )

        # Create the ChatOpenAI client
        if llm_config.api_type == APITypeEnum.RESPONSES:
            client = ChatOpenAI(stream_usage=True, use_responses_api=True, use_previous_response_id=True, **config_dict)
        else:
            client = ChatOpenAI(stream_usage=True, **config_dict)

        yield _patch_llm_based_on_config(client, llm_config)
    finally:
        # Ensure the httpx client is properly closed to avoid resource leaks
        if http_async_client is not None:
            await http_async_client.aclose()
```

### Step 4: Run existing tests to ensure no regressions

Run: `pytest tests/nat/plugins/langchain/ -v -k dynamo`
Expected: PASS

### Step 5: Commit

```bash
git add packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py tests/nat/plugins/langchain/test_dynamo_trie_loading.py
git commit --signoff -m "feat(langchain): load prediction trie in dynamo_langchain

Loads prediction trie from prediction_trie_path config and passes
PredictionTrieLookup to httpx client for dynamic header injection."
```

---

## Task 7: End-to-End Integration Test

**Files:**
- Test: `tests/nat/llm/test_runtime_prediction_e2e.py`

### Step 1: Write end-to-end integration test

```python
# tests/nat/llm/test_runtime_prediction_e2e.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for runtime prediction trie integration."""

import tempfile
from pathlib import Path

import pytest

from nat.builder.context import Context
from nat.data_models.intermediate_step import IntermediateStepPayload
from nat.data_models.intermediate_step import IntermediateStepType
from nat.llm.dynamo_llm import _create_dynamic_prediction_hook
from nat.llm.prediction_context import get_call_tracker
from nat.profiler.prediction_trie import load_prediction_trie
from nat.profiler.prediction_trie import PredictionTrieLookup
from nat.profiler.prediction_trie import save_prediction_trie
from nat.profiler.prediction_trie.data_models import LLMCallPrediction
from nat.profiler.prediction_trie.data_models import PredictionMetrics
from nat.profiler.prediction_trie.data_models import PredictionTrieNode


class MockRequest:
    """Mock httpx.Request for testing."""

    def __init__(self):
        self.headers = {}


def create_test_trie() -> PredictionTrieNode:
    """Create a test trie with known predictions."""
    # Agent at call 1: 2 remaining, 500ms interarrival, 150 tokens
    call_1_prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=2.0, p50=2.0, p90=3.0, p95=4.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=500.0, p50=450.0, p90=700.0, p95=800.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=150.0, p50=140.0, p90=200.0, p95=250.0),
    )

    # Agent at call 2: 1 remaining, 300ms interarrival, 100 tokens
    call_2_prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=1.0, p50=1.0, p90=2.0, p95=2.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=300.0, p50=280.0, p90=400.0, p95=450.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=100.0, p50=90.0, p90=150.0, p95=180.0),
    )

    # Agent at call 3: 0 remaining
    call_3_prediction = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=10, mean=0.0, p50=0.0, p90=0.0, p95=0.0),
        interarrival_ms=PredictionMetrics(sample_count=10, mean=0.0, p50=0.0, p90=0.0, p95=0.0),
        output_tokens=PredictionMetrics(sample_count=10, mean=80.0, p50=75.0, p90=120.0, p95=140.0),
    )

    # Aggregated for fallback
    aggregated = LLMCallPrediction(
        remaining_calls=PredictionMetrics(sample_count=30, mean=1.0, p50=1.0, p90=2.0, p95=3.0),
        interarrival_ms=PredictionMetrics(sample_count=30, mean=400.0, p50=380.0, p90=550.0, p95=600.0),
        output_tokens=PredictionMetrics(sample_count=30, mean=110.0, p50=100.0, p90=160.0, p95=190.0),
    )

    agent_node = PredictionTrieNode(
        name="react_agent",
        predictions_by_call_index={1: call_1_prediction, 2: call_2_prediction, 3: call_3_prediction},
        predictions_any_index=aggregated,
    )

    workflow_node = PredictionTrieNode(
        name="my_workflow",
        children={"react_agent": agent_node},
        predictions_any_index=aggregated,
    )

    return PredictionTrieNode(
        name="root",
        children={"my_workflow": workflow_node},
        predictions_any_index=aggregated,
    )


async def test_e2e_prediction_headers_injected_correctly():
    """Test complete flow: context tracking -> step manager -> hook -> headers."""
    # Create and save trie
    trie = create_test_trie()

    with tempfile.TemporaryDirectory() as tmpdir:
        trie_path = Path(tmpdir) / "prediction_trie.json"
        save_prediction_trie(trie, trie_path, workflow_name="test")

        # Load trie
        loaded_trie = load_prediction_trie(trie_path)
        lookup = PredictionTrieLookup(loaded_trie)

        # Create hook
        hook = _create_dynamic_prediction_hook(lookup)

        ctx = Context.get()
        state = ctx._context_state
        step_manager = ctx.intermediate_step_manager

        # Reset state
        state._function_path_stack.set(None)

        with ctx.push_active_function("my_workflow", input_data=None):
            with ctx.push_active_function("react_agent", input_data=None):
                # Simulate first LLM call
                step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID="llm-1",
                        event_type=IntermediateStepType.LLM_START,
                        name="test-model",
                    )
                )

                request1 = MockRequest()
                await hook(request1)

                # Should have call 1 predictions: 2 remaining
                assert request1.headers["x-nat-remaining-llm-calls"] == "2"
                assert request1.headers["x-nat-interarrival-ms"] == "500"
                assert request1.headers["x-nat-expected-output-tokens"] == "200"

                # Simulate second LLM call
                step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID="llm-2",
                        event_type=IntermediateStepType.LLM_START,
                        name="test-model",
                    )
                )

                request2 = MockRequest()
                await hook(request2)

                # Should have call 2 predictions: 1 remaining
                assert request2.headers["x-nat-remaining-llm-calls"] == "1"
                assert request2.headers["x-nat-interarrival-ms"] == "300"
                assert request2.headers["x-nat-expected-output-tokens"] == "150"

                # Simulate third LLM call
                step_manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID="llm-3",
                        event_type=IntermediateStepType.LLM_START,
                        name="test-model",
                    )
                )

                request3 = MockRequest()
                await hook(request3)

                # Should have call 3 predictions: 0 remaining
                assert request3.headers["x-nat-remaining-llm-calls"] == "0"
                assert request3.headers["x-nat-expected-output-tokens"] == "120"


async def test_e2e_fallback_to_root():
    """Test that unknown paths fall back to root predictions."""
    trie = create_test_trie()
    lookup = PredictionTrieLookup(trie)
    hook = _create_dynamic_prediction_hook(lookup)

    ctx = Context.get()
    state = ctx._context_state
    step_manager = ctx.intermediate_step_manager

    # Reset state
    state._function_path_stack.set(None)

    with ctx.push_active_function("unknown_workflow", input_data=None):
        step_manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID="llm-unknown",
                event_type=IntermediateStepType.LLM_START,
                name="test-model",
            )
        )

        request = MockRequest()
        await hook(request)

        # Should fall back to root aggregated predictions
        assert "x-nat-remaining-llm-calls" in request.headers
        assert request.headers["x-nat-remaining-llm-calls"] == "1"  # aggregated mean
```

### Step 2: Run e2e test

Run: `pytest tests/nat/llm/test_runtime_prediction_e2e.py -v`
Expected: PASS

### Step 3: Commit

```bash
git add tests/nat/llm/test_runtime_prediction_e2e.py
git commit --signoff -m "test: add end-to-end runtime prediction trie test

Validates complete flow: function path tracking -> call tracker increment
-> dynamic hook lookup -> correct headers injected for each call index."
```

---

## Summary

This plan implements runtime prediction trie integration in 7 tasks:

1. **Function Path Stack** - Add ContextVar to ContextState
2. **Path Tracking** - Update push_active_function to track path
3. **Call Tracker Integration** - Increment tracker in IntermediateStepManager on LLM_START
4. **Dynamic Hook** - Create hook that reads context and looks up predictions
5. **Client Update** - Add prediction_lookup param to client creation
6. **LangChain Integration** - Load trie in dynamo_langchain
7. **E2E Test** - Validate complete flow

Each task follows TDD: write failing test, implement, verify, commit.
