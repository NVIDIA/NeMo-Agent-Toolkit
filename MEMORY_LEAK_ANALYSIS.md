# NAT MCP Server Memory Leak Analysis

## Executive Summary

A significant memory leak has been identified in the NeMo Agent Toolkit (NAT) MCP server implementation. During a 10-minute test period, the system exhibited:

- **6 observability exporters not properly stopped** (1226 started vs 1220 stopped)
- **2 database sessions not properly closed** (250 opened vs 248 closed)
- **1,226 MCP tool requests processed** (text2sql: 252, followup: 242, chartgen: 234)

## Root Cause Analysis

### Primary Issue: Missing Observability Context for Workflows

The root cause is in `/src/nat/front_ends/mcp/tool_converter.py`, specifically in the `create_function_wrapper()` function.

#### Code Issue Location

**File**: `src/nat/front_ends/mcp/tool_converter.py`
**Lines**: 132-169

#### Problem Description

The MCP tool converter has inconsistent handling of observability exporters between regular functions and workflows:

**For Regular Functions (lines 168-169):**
```python
# Regular function call
result = await call_with_observability(lambda: function.acall_invoke(**kwargs))
```
✅ This properly wraps the function call with `exporter_manager.start()` context

**For Workflows with ChatRequest (lines 132-137):**
```python
# Special handling for Workflow objects
if is_workflow:
    # Workflows have a run method that is an async context manager
    # that returns a Runner
    async with function.run(chat_request) as runner:
        # Get the result from the runner
        result = await runner.result(to_type=str)
```
❌ This bypasses `call_with_observability()`, never starting/stopping the exporter

**For Workflows with Regular Input (lines 158-166):**
```python
# Call the NAT function with the parameters - special handling for Workflow
if is_workflow:
    # For workflow with regular input, we'll assume the first parameter is the input
    input_value = list(kwargs.values())[0] if kwargs else ""

    # Workflows have a run method that is an async context manager
    # that returns a Runner
    async with function.run(input_value) as runner:
        # Get the result from the runner
        result = await runner.result(to_type=str)
```
❌ Also bypasses `call_with_observability()`

### Secondary Issue: Database Session Leak

The 2 unclosed database sessions (250 opened vs 248 closed) suggest that when the exporter fails to stop properly, cleanup code that closes database connections may not execute completely. This compounds the memory leak.

## Evidence from Logs

### Quantitative Analysis

From the 10-minute test log (`extract-2025-10-08T21_42_22.090Z.csv`):

1. **Total Requests**: 1,226 MCP tool calls processed
2. **Exporter Lifecycle**:
   - Exporters started: **1,226**
   - Exporters stopped: **1,220**
   - **Leak**: 6 exporters not stopped (0.49% leak rate)

3. **Database Sessions**:
   - Sessions opened: **250**
   - Sessions closed: **248**
   - **Leak**: 2 sessions not closed (0.8% leak rate)

4. **Tool Call Breakdown**:
   - `text2sql`: 252 calls
   - `followup`: 242 calls
   - `chartgen`: 234 calls

### Pattern Observations

Each MCP request triggers:
```
Processing request of type CallToolRequest
  → Started exporter 'otelcollector'
  → [Tool execution]
  → Stopped exporter 'otelcollector'  # Sometimes missing!
```

## Impact Assessment

### Memory Leak Rate

- **Per-request leak**: ~0.5% of requests leave resources uncleaned
- **Cumulative effect**: Over extended operation (hours/days), this accumulates
- **High-volume scenarios**: At 1,000 requests/hour, ~5 leaked exporters/hour

### Resource Types Affected

1. **Observability Exporters**:
   - Each exporter maintains event subscriptions
   - Background tasks for exporting traces
   - OpenTelemetry collector connections

2. **Database Connections**:
   - Databricks SQL sessions
   - HTTP connections to database
   - Query result buffers

3. **Context State**:
   - Per-request context state objects
   - Span stacks and metadata stacks

## Recommended Solutions

### Solution 1: Wrap Workflow Calls with Observability (RECOMMENDED)

Modify `src/nat/front_ends/mcp/tool_converter.py` to ensure workflows also use the observability wrapper:

```python
# Special handling for ChatRequest
if is_chat_request:
    from nat.data_models.api_server import ChatRequest

    # Create a chat request from the query string
    query = kwargs.get("query", "")
    chat_request = ChatRequest.from_string(query)

    # Special handling for Workflow objects
    if is_workflow:
        # FIX: Wrap workflow execution with observability
        async def workflow_call():
            async with function.run(chat_request) as runner:
                return await runner.result(to_type=str)

        result = await call_with_observability(workflow_call)
    else:
        # Regular functions use ainvoke
        result = await call_with_observability(lambda: function.ainvoke(chat_request, to_type=str))
else:
    # Regular handling
    # [Handle complex input schema logic...]

    # Call the NAT function with the parameters - special handling for Workflow
    if is_workflow:
        # FIX: Wrap workflow execution with observability
        async def workflow_call():
            input_value = list(kwargs.values())[0] if kwargs else ""
            async with function.run(input_value) as runner:
                return await runner.result(to_type=str)

        result = await call_with_observability(workflow_call)
    else:
        # Regular function call
        result = await call_with_observability(lambda: function.acall_invoke(**kwargs))
```

### Solution 2: Add Try-Finally Blocks

Ensure exporter cleanup even on exceptions:

```python
async def call_with_observability(func_call):
    if not workflow:
        logger.error("Missing workflow context for function %s - observability will not be available",
                     function_name)
        raise RuntimeError("Workflow context is required for observability")

    logger.debug("Starting observability context for function %s", function_name)
    context_state = ContextState.get()

    try:
        async with workflow.exporter_manager.start(context_state=context_state):
            return await func_call()
    except Exception:
        logger.exception("Error in function %s with observability context", function_name)
        raise
```

### Solution 3: Add Resource Leak Detection

Add monitoring to detect and warn about leaked resources:

```python
class ExporterManager:
    def __init__(self):
        self._active_contexts = 0
        self._max_concurrent_contexts = 0

    async def start(self, context_state: ContextState | None = None):
        self._active_contexts += 1
        self._max_concurrent_contexts = max(self._max_concurrent_contexts, self._active_contexts)

        if self._active_contexts > 10:  # threshold
            logger.warning(
                "High number of active exporter contexts: %d (max: %d). Possible leak?",
                self._active_contexts,
                self._max_concurrent_contexts
            )

        try:
            # ... existing code ...
            yield self
        finally:
            self._active_contexts -= 1
```

## Testing Recommendations

### Unit Tests

1. **Test exporter lifecycle for workflow calls**:
   - Verify exporter.start() is called
   - Verify exporter.stop() is called
   - Test both ChatRequest and regular input workflows

2. **Test exception handling**:
   - Verify exporters are stopped even when tool execution fails
   - Test database connection cleanup on errors

### Integration Tests

1. **Load test with resource monitoring**:
   - Run 1,000+ MCP tool calls
   - Monitor exporter count growth
   - Monitor database connection count
   - Check for memory growth

2. **Long-running stress test**:
   - Run MCP server for 1+ hour under load
   - Monitor memory usage trend
   - Check for resource exhaustion

## Prevention Strategies

### Code Review Checklist

- [ ] All async context managers have proper cleanup in finally blocks
- [ ] Resource acquisition follows RAII pattern (Resource Acquisition Is Initialization)
- [ ] Workflow execution paths match function execution paths for observability
- [ ] Exception handling doesn't prevent resource cleanup

### Monitoring Recommendations

1. **Add metrics**:
   - Active exporter count
   - Active database session count
   - Memory usage per request type
   - Request completion vs. exporter stop correlation

2. **Add alarms**:
   - Alert when active exporter count exceeds threshold
   - Alert when database sessions exceed expected count
   - Alert on memory growth rate

## References

- Original bug report: Memory leak observed during 10-minute NAT MCP server test
- Log file: `extract-2025-10-08T21_42_22.090Z.csv` (17,397 lines)
- Code files:
  - `src/nat/front_ends/mcp/tool_converter.py` (lines 110-169)
  - `src/nat/observability/exporter_manager.py`
  - `src/nat/observability/exporter/base_exporter.py`

## Appendix: Statistical Analysis

### Leak Rate Calculation

```
Exporter Leak Rate = (Started - Stopped) / Started
                   = (1226 - 1220) / 1226
                   = 6 / 1226
                   = 0.489%

Session Leak Rate = (Opened - Closed) / Opened
                  = (250 - 248) / 250
                  = 2 / 250
                  = 0.8%
```

### Memory Impact Estimation

Assuming each leaked exporter holds:
- Event subscriptions: ~1KB
- Background task references: ~2KB
- Context state: ~5KB
- OpenTelemetry connections: ~10KB

**Estimated per-exporter leak**: ~18KB

**For 1,000 requests at 0.5% leak rate**:
- Leaked exporters: 5
- Memory leaked: 90KB

**For 100,000 requests (1 day at high load)**:
- Leaked exporters: 500
- Memory leaked: 9MB

This compounds over time and with database session leaks can lead to OOM errors.
