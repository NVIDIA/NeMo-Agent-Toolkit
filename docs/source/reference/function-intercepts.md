<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Function Intercepts

## Overview

Function intercepts provide a powerful mechanism to add cross-cutting concerns and preprocessing logic to functions in the NeMo Agent toolkit without modifying the function implementation itself. Intercepts execute before a function's core logic, enabling capabilities such as caching, logging, authentication, rate limiting, and other middleware-like behaviors.

### Key Concepts

**Function Intercept**: A callable that runs before a function's `ainvoke` or `astream` methods, with the ability to:
- Inspect and modify function inputs
- Execute preprocessing logic
- Short-circuit execution by returning cached or computed results
- Delegate to the next intercept or the function itself
- Transform or augment function outputs

**Intercept Chain**: A sequence of intercepts that execute in order before reaching the function. Each intercept can delegate to the next one in the chain.

**Final Intercept**: A special intercept that terminates the chain. Only one final intercept is allowed per function, and it must be the last intercept in the chain.

### When to Use Function Intercepts

Function intercepts are ideal for implementing:

- **Caching**: Store and retrieve function results to avoid redundant computation
- **Authentication and Authorization**: Verify user permissions before function execution
- **Rate Limiting**: Control the frequency of function invocations
- **Logging and Monitoring**: Track function calls and performance metrics
- **Input Validation**: Verify inputs meet requirements before processing
- **Retry Logic**: Automatically retry failed function calls
- **Request Transformation**: Modify inputs before they reach the function
- **Response Transformation**: Modify outputs before they're returned

## How Function Intercepts Work

### Intercept Chain Architecture

Function intercepts form a chain of responsibility pattern where each intercept can:

1. **Inspect** the input before passing it forward
2. **Modify** the input before delegation
3. **Execute** custom logic (logging, validation, and so on)
4. **Short-circuit** by returning a value without calling the next intercept
5. **Delegate** to the next intercept in the chain
6. **Transform** the result from downstream intercepts

```
┌─────────────────────────────────────────────────────────┐
│ Function Call                                           │
└────────────┬────────────────────────────────────────────┘
             │
             ▼
      ┌─────────────┐
      │ Intercept 1 │  (such as, logging)
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │ Intercept 2 │  (such as, validation)
      └──────┬──────┘
             │
             ▼
      ┌─────────────┐
      │ Intercept 3 │  (such as, caching - final)
      └──────┬──────┘
             │
             ▼
    ┌────────────────┐
    │ Actual Function│
    └────────────────┘
```

### Execution Flow

1. **Registration**: Intercepts are registered with a function at build time
2. **Chain Construction**: The builder creates an intercept chain when building the workflow
3. **Invocation**: When the function is called, the chain executes from first to last
4. **Delegation**: Each intercept can delegate to the next using the `next_call` parameter
5. **Return**: Results flow back through the chain to the caller

### Final Intercepts

A final intercept is marked with `is_final=True` and has special properties:

- **Must be last**: Final intercepts must be the last in the chain
- **Only one allowed**: Only one final intercept is permitted per function
- **Termination**: By default, final intercepts do not delegate to the function unless explicitly implemented

Final intercepts are useful for behaviors like caching where you want to potentially bypass the function entirely.

## Creating Custom Function Intercepts

### Basic Intercept Implementation

All intercepts must inherit from the `FunctionIntercept` base class:

```python
from nat.intercepts import FunctionIntercept, FunctionInterceptContext
from nat.intercepts import SingleInvokeCallable, StreamInvokeCallable
from typing import Any
from collections.abc import AsyncIterator


class MyCustomIntercept(FunctionIntercept):
    """A custom intercept that logs function calls."""
    
    def __init__(self, *, log_level: str = "INFO"):
        super().__init__(is_final=False)
        self.log_level = log_level
    
    async def intercept_invoke(
        self,
        value: Any,
        next_call: SingleInvokeCallable,
        context: FunctionInterceptContext
    ) -> Any:
        """Intercept single-output invocations."""
        print(f"[{self.log_level}] Calling function: {context.name}")
        print(f"[{self.log_level}] Input: {value}")
        
        # Delegate to the next intercept or function
        result = await next_call(value)
        
        print(f"[{self.log_level}] Result: {result}")
        return result
    
    async def intercept_stream(
        self,
        value: Any,
        next_call: StreamInvokeCallable,
        context: FunctionInterceptContext
    ) -> AsyncIterator[Any]:
        """Intercept streaming invocations."""
        print(f"[{self.log_level}] Streaming call to: {context.name}")
        
        # Delegate to the next intercept or function
        async for chunk in next_call(value):
            print(f"[{self.log_level}] Chunk: {chunk}")
            yield chunk
```

### Intercept Context

The `FunctionInterceptContext` provides metadata about the function being intercepted:

```python
@dataclasses.dataclass(frozen=True)
class FunctionInterceptContext:
    """Context information supplied to each intercept."""
    
    name: str                               # Function name
    config: FunctionBaseConfig              # Function configuration
    description: str | None                 # Function description
    input_schema: type[BaseModel]           # Input schema
    single_output_schema: type[BaseModel] | type[None]  # Single output schema
    stream_output_schema: type[BaseModel] | type[None]  # Stream output schema
```

### Implementing a Final Intercept

Here's an example of a validation intercept that short-circuits on invalid input:

```python
from pydantic import ValidationError


class ValidationIntercept(FunctionIntercept):
    """A final intercept that validates inputs."""
    
    def __init__(self, *, strict: bool = True):
        super().__init__(is_final=True)
        self.strict = strict
    
    async def intercept_invoke(
        self,
        value: Any,
        next_call: SingleInvokeCallable,
        context: FunctionInterceptContext
    ) -> Any:
        """Validate input before delegating."""
        try:
            # Validate against the input schema
            validated = context.input_schema.model_validate(value)
        except ValidationError as e:
            if self.strict:
                raise ValueError(f"Invalid input for {context.name}: {e}")
            else:
                # In non-strict mode, try to proceed anyway
                return await next_call(value)
        
        # Delegate to the function with validated input
        return await next_call(validated)
```

## Using Function Intercepts

### Registering Intercepts with Functions

Intercepts are configured when registering functions using the `@register_function` decorator:

```python
from nat.cli.register_workflow import register_function
from nat.intercepts import CacheIntercept
from nat.data_models.function import FunctionBaseConfig


class MyFunctionConfig(FunctionBaseConfig, name="my_function"):
    use_cache: bool = True


@register_function(
    config_type=MyFunctionConfig,
    intercepts=[CacheIntercept(enabled_mode="always", similarity_threshold=1.0)]
)
async def my_function(config: MyFunctionConfig, builder: Builder):
    """A function with caching enabled."""
    
    async def process(input_data: dict) -> dict:
        # Expensive computation here
        return {"result": input_data["value"] * 2}
    
    from nat.builder.function import LambdaFunction
    from nat.builder.function_info import FunctionInfo
    
    function_info = FunctionInfo.from_fn(
        process,
        description="Process input data"
    )
    
    yield LambdaFunction.from_info(
        config=config,
        info=function_info
    )
```

### Chaining Multiple Intercepts

You can chain multiple intercepts together. They execute in the order provided:

```python
from nat.intercepts import CacheIntercept


class LoggingIntercept(FunctionIntercept):
    """Log function calls."""
    
    async def intercept_invoke(self, value, next_call, context):
        logger.info(f"Calling {context.name} with {value}")
        result = await next_call(value)
        logger.info(f"Function {context.name} returned {result}")
        return result


@register_function(
    config_type=MyFunctionConfig,
    intercepts=[
        LoggingIntercept(),              # Executes first
        ValidationIntercept(strict=True),  # Executes second
        CacheIntercept(
            enabled_mode="eval",          # Executes last (final)
            similarity_threshold=0.95
        )
    ]
)
async def my_function(config: MyFunctionConfig, builder: Builder):
    # Function implementation
    ...
```

### Configuration-Driven Intercepts

You can make intercepts configurable through the function's configuration:

```python
class MyFunctionConfig(FunctionBaseConfig, name="my_function"):
    enable_caching: bool = True
    cache_threshold: float = 1.0
    enable_logging: bool = False


@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):
    """A function with conditional intercepts."""
    
    # Build intercepts based on configuration
    intercepts = []
    
    if config.enable_logging:
        intercepts.append(LoggingIntercept())
    
    if config.enable_caching:
        intercepts.append(
            CacheIntercept(
                enabled_mode="always",
                similarity_threshold=config.cache_threshold
            )
        )
    
    # Create function with dynamic intercepts
    # (Implementation depends on how you construct your function)
    ...
```

## Built-in Intercepts

### Cache Intercept

The `CacheIntercept` is a built-in final intercept that caches function outputs based on input similarity. It's particularly useful for expensive computations or API calls.

#### Features

- **Input Serialization**: Automatically serializes function inputs to strings for comparison
- **Similarity Matching**: Supports both exact matching and fuzzy similarity matching
- **Evaluation Mode**: Can be configured to cache only during evaluation
- **Streaming Bypass**: Automatically bypasses caching for streaming calls
- **Fast Exact Matching**: Uses optimized exact string matching when threshold is 1.0

#### Basic Usage

```python
from nat.intercepts import CacheIntercept

# Create a cache intercept with exact matching
cache = CacheIntercept(
    enabled_mode="always",      # Always cache
    similarity_threshold=1.0     # Exact match only
)

# Use with function registration
@register_function(
    config_type=MyFunctionConfig,
    intercepts=[cache]
)
async def expensive_function(config, builder):
    # Expensive function that benefits from caching
    ...
```

#### Parameters

**enabled_mode** (str, default: "eval")
- `"always"`: Cache is always active
- `"eval"`: Cache is active only when `Context.is_evaluating` is `True`

**similarity_threshold** (float, default: 1.0)
- Range: 0.0 to 1.0
- `1.0`: Exact string matching (fastest, recommended for most cases)
- `< 1.0`: Fuzzy matching using Python's `difflib.SequenceMatcher`
- Lower values allow more flexibility but may cache overly similar inputs

#### Configuration Examples

**Always Cache with Exact Matching**

```python
# Best for deterministic functions where inputs must match exactly
cache = CacheIntercept(
    enabled_mode="always",
    similarity_threshold=1.0
)
```

**Cache Only During Evaluation**

```python
# Useful when you want caching during testing but not in production
cache = CacheIntercept(
    enabled_mode="eval",
    similarity_threshold=1.0
)
```

**Fuzzy Matching for Similar Inputs**

```python
# Allows caching for inputs that are 90% similar
# Use with caution as this may cache different logical inputs
cache = CacheIntercept(
    enabled_mode="always",
    similarity_threshold=0.9
)
```

#### Practical Examples

**Caching API Responses**

```python
from pydantic import BaseModel


class APIInput(BaseModel):
    query: str
    params: dict


class APIOutput(BaseModel):
    response: dict
    status: int


@register_function(
    config_type=APIFunctionConfig,
    intercepts=[
        CacheIntercept(
            enabled_mode="always",
            similarity_threshold=1.0
        )
    ]
)
async def call_external_api(config, builder):
    """Call external API with caching."""
    
    async def api_call(input_data: APIInput) -> APIOutput:
        # Expensive API call
        response = await external_service.query(
            input_data.query,
            **input_data.params
        )
        return APIOutput(
            response=response,
            status=200
        )
    
    # Create and return function
    ...
```

**Caching Database Queries**

```python
@register_function(
    config_type=DatabaseFunctionConfig,
    intercepts=[
        CacheIntercept(
            enabled_mode="eval",  # Only cache during testing
            similarity_threshold=1.0
        )
    ]
)
async def query_database(config, builder):
    """Query database with evaluation-time caching."""
    
    async def execute_query(query: str) -> list[dict]:
        # Expensive database query
        return await db.execute(query)
    
    # Create and return function
    ...
```

**Caching LLM Responses**

```python
@register_function(
    config_type=LLMFunctionConfig,
    intercepts=[
        CacheIntercept(
            enabled_mode="eval",
            similarity_threshold=0.95  # Allow slight prompt variations
        )
    ]
)
async def llm_completion(config, builder):
    """LLM completion with fuzzy caching."""
    
    async def complete(prompt: str) -> str:
        # Expensive LLM call
        return await llm.complete(prompt)
    
    # Create and return function
    ...
```

#### Behavior and Limitations

**Serialization Failures**

If the cache intercept cannot serialize an input, it gracefully falls back to calling the function:

```python
# This will bypass cache if serialization fails
result = await function(non_serializable_object)
```

**Streaming Calls**

The cache intercept always bypasses caching for streaming calls to avoid buffering entire streams in memory:

```python
# Streaming calls are never cached
async for chunk in function.astream(input_data):
    process(chunk)
```

**Memory Considerations**

The cache intercept stores results in memory. For functions with many unique inputs or large outputs, consider:
- Using shorter cache lifetimes
- Implementing cache eviction policies
- Using external caching solutions for production

#### Advanced Usage with Context

The cache intercept can access the workflow context to make caching decisions:

```python
from nat.builder.context import Context, ContextState


# In your workflow
context_state = ContextState.get()
context = Context(context_state)

# Set evaluation mode
context_state.is_evaluating.set(True)

# Now functions with eval mode caching will cache results
result = await function(input_data)
```

## Best Practices

### Design Principles

1. **Keep Intercepts Focused**: Each intercept should have a single, well-defined responsibility
2. **Preserve Function Contracts**: Intercepts should not change input/output schemas
3. **Handle Errors Gracefully**: Intercepts should fail gracefully and log errors appropriately
4. **Document Side Effects**: Clearly document any side effects (logging, caching, and so on)
5. **Consider Performance**: Intercepts add overhead; keep them lightweight

### Intercept Ordering

Order matters when chaining intercepts. Consider this recommended order:

1. **Logging/Monitoring**: First to capture all calls
2. **Authentication/Authorization**: Early to reject unauthorized calls
3. **Validation**: Validate inputs before expensive operations
4. **Rate Limiting**: Prevent excessive calls
5. **Caching**: Final intercept to potentially skip execution

```python
intercepts=[
    LoggingIntercept(),           # 1. Log everything
    AuthenticationIntercept(),    # 2. Check permissions
    ValidationIntercept(),        # 3. Validate inputs
    RateLimitIntercept(),        # 4. Check rate limits
    CacheIntercept(...)          # 5. Cache results (final)
]
```

### Testing Intercepts

Always test intercepts independently and as part of the chain:

```python
import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_cache_intercept():
    """Test cache intercept behavior."""
    cache = CacheIntercept(enabled_mode="always", similarity_threshold=1.0)
    
    # Create mock context
    context = FunctionInterceptContext(
        name="test_function",
        config=MockConfig(),
        description="Test function",
        input_schema=dict,
        single_output_schema=dict,
        stream_output_schema=None
    )
    
    # Create mock function
    call_count = 0
    async def mock_function(value):
        nonlocal call_count
        call_count += 1
        return {"result": value * 2}
    
    # First call should invoke function
    input1 = {"value": 5}
    result1 = await cache.intercept_invoke(input1, mock_function, context)
    assert call_count == 1
    assert result1 == {"result": 10}
    
    # Second call should use cache
    result2 = await cache.intercept_invoke(input1, mock_function, context)
    assert call_count == 1  # Not called again
    assert result2 == {"result": 10}
```

### Performance Considerations

1. **Minimize Intercept Overhead**: Keep intercept logic lightweight
2. **Async-First**: Use async operations in intercepts to avoid blocking
3. **Cache Wisely**: Only cache when the computation is expensive
4. **Monitor Performance**: Track intercept impact on latency

### Security Considerations

1. **Validate Inputs**: Use intercepts to validate and sanitize inputs
2. **Authentication**: Implement authentication intercepts for sensitive functions
3. **Audit Logging**: Log security-relevant events in intercepts
4. **Rate Limiting**: Prevent abuse with rate-limiting intercepts

## Common Patterns

### Conditional Execution

```python
class ConditionalIntercept(FunctionIntercept):
    """Execute logic only when condition is met."""
    
    def __init__(self, *, condition: Callable[[], bool]):
        super().__init__()
        self.condition = condition
    
    async def intercept_invoke(self, value, next_call, context):
        if self.condition():
            # Execute conditional logic
            pass
        return await next_call(value)
```

### Input Transformation

```python
class TransformIntercept(FunctionIntercept):
    """Transform inputs before function execution."""
    
    async def intercept_invoke(self, value, next_call, context):
        # Transform input
        transformed = self.transform(value)
        
        # Call with transformed input
        result = await next_call(transformed)
        
        # Optionally transform output
        return self.transform_output(result)
```

### Error Handling and Retry

```python
class RetryIntercept(FunctionIntercept):
    """Retry failed function calls."""
    
    def __init__(self, *, max_retries: int = 3, backoff: float = 1.0):
        super().__init__()
        self.max_retries = max_retries
        self.backoff = backoff
    
    async def intercept_invoke(self, value, next_call, context):
        for attempt in range(self.max_retries):
            try:
                return await next_call(value)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.backoff * (2 ** attempt))
```

## Troubleshooting

### Common Issues

**Intercept Not Executing**

- Verify intercept is registered with the function
- Check intercept order in the chain
- Ensure intercept methods are properly implemented

**Cache Not Working**

- Check `enabled_mode` setting
- Verify `Context.is_evaluating` is set correctly for eval mode
- Ensure inputs are serializable
- Check similarity threshold setting

**Performance Degradation**

- Profile intercepts to find bottlenecks
- Consider making intercepts async
- Reduce logging verbosity
- Optimize caching strategy

### Debugging Intercepts

Enable detailed logging to troubleshoot intercept behavior:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("nat.intercepts")
logger.setLevel(logging.DEBUG)
```

## API Reference

For detailed API documentation, refer to:

- {py:class}`~nat.intercepts.function_intercept.FunctionIntercept`: Base intercept class
- {py:class}`~nat.intercepts.function_intercept.FunctionInterceptContext`: Context information
- {py:class}`~nat.intercepts.function_intercept.FunctionInterceptChain`: Chain management
- {py:class}`~nat.intercepts.cache_intercept.CacheIntercept`: Built-in cache intercept
- {py:func}`~nat.intercepts.function_intercept.validate_intercepts`: Validation utility

## See Also

- [Writing Custom Functions](../extend/functions.md): Guide to creating custom functions
- [Function Groups](../extend/function-groups.md): Organizing related functions
- [Plugin System](../extend/plugins.md): Extending the NeMo Agent toolkit

