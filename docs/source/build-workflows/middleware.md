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

# Middleware

## Overview

Middleware provides a powerful mechanism for adding cross-cutting concerns to functions in the NeMo Agent Toolkit without modifying the function implementation itself. Like middleware in web frameworks (Express.js, FastAPI, etc.), middleware wraps function calls with a four-phase pattern:

1. **Preprocess** - Inspect and modify inputs before calling next
2. **Call Next** - Delegate to the next middleware or function
3. **Postprocess** - Process, transform, or augment outputs
4. **Continue** - Return or yield the final result

Middleware components are first-class components in NAT, configured in YAML and built by the workflow builder, just like retrievers, memory providers, and other components.

## Key Concepts

**Middleware Component**: A middleware component that:
- Is configured in YAML with a `middleware` section
- Is built by the workflow builder before functions and function groups
- Wraps a function's `ainvoke` or `astream` methods
- Can be applied to individual functions or entire function groups
- Can preprocess inputs, postprocess outputs, or short-circuit execution

**Middleware Chain**: A sequence of middleware that execute in order, forming an "onion" structure where control flows in through preprocessing, down to the function, and back out through postprocessing.

**Final Middleware**: A special middleware marked with `is_final=True` that can terminate the chain. Only one final middleware is allowed per function, and it must be the last in the chain.

## Component-Based Architecture

Middleware follows the same component pattern as other NAT components:

```yaml
middleware:
  my_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

  my_logger:
    _type: logging_middleware
    log_level: INFO

functions:
  my_function:
    _type: my_function_type
    middleware: ["my_logger", "my_cache"]  # Apply middleware in order
    # Other function config...

function_groups:
  my_function_group:
    _type: my_function_group_type
    middleware: ["my_logger", "my_cache"]  # Apply middleware to all functions in the group
    # Other function group config...
```

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config, builder):
    # Function implementation
    ...
```

## Creating Custom Function Middleware

### Step 1: Define the Configuration

Create a configuration class inheriting from `FunctionMiddlewareBaseConfig`:

```python
from pydantic import Field
from nat.data_models.middleware import FunctionMiddlewareBaseConfig


class LoggingMiddlewareConfig(FunctionMiddlewareBaseConfig, name="logging_middleware"):
  """Configuration for logging middleware."""

  log_level: str = Field(
    default="INFO",
    description="Logging level (DEBUG, INFO, WARNING, ERROR)"
  )
  include_inputs: bool = Field(
    default=True,
    description="Whether to log function inputs"
  )
  include_outputs: bool = Field(
    default=True,
    description="Whether to log function outputs"
  )
```

### Step 2: Implement the Middleware Class

Create the middleware class inheriting from `FunctionMiddleware`:

```python
from nat.middleware import FunctionMiddleware, FunctionMiddlewareContext
from nat.middleware import CallNext, CallNextStream
import logging
from typing import Any
from collections.abc import AsyncIterator


class LoggingMiddleware(FunctionMiddleware):
    """Logging middleware that tracks function calls."""

    def __init__(self, *, log_level: str, include_inputs: bool, include_outputs: bool):
        super().__init__(is_final=False)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.include_inputs = include_inputs
        self.include_outputs = include_outputs

    async def function_middleware_invoke(
        self,
        value: Any,
        call_next: CallNext,
        context: FunctionMiddlewareContext
    ) -> Any:
        """Middleware for single-output invocations."""
        # Phase 1: Preprocess
        if self.include_inputs:
            self.logger.info(f"Calling {context.name} with input: {value}")

        # Phase 2: Call next
        result = await call_next(value)

        # Phase 3: Postprocess
        if self.include_outputs:
            self.logger.info(f"Function {context.name} returned: {result}")

        # Phase 4: Continue
        return result

    async def function_middleware_stream(
        self,
        value: Any,
        call_next: CallNextStream,
        context: FunctionMiddlewareContext
    ) -> AsyncIterator[Any]:
        """Middleware for streaming invocations."""
        # Phase 1: Preprocess
        if self.include_inputs:
            self.logger.info(f"Streaming call to {context.name} with input: {value}")

        # Phase 2-3: Call next and yield chunks
        chunk_count = 0
        async for chunk in call_next(value):
            chunk_count += 1
            yield chunk

        # Phase 4: Cleanup
        if self.include_outputs:
            self.logger.info(f"Streamed {chunk_count} chunks from {context.name}")
```

### Step 3: Register the Component

Create a registration module following the idiomatic pattern:

```python
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_middleware
from .logging_middleware import LoggingMiddleware, LoggingMiddlewareConfig


@register_middleware(config_type=LoggingMiddlewareConfig)
async def logging_middleware(config: LoggingMiddlewareConfig, builder: Builder):
    """Build logging middleware from configuration.

    Args:
        config: The logging middleware configuration
        builder: The workflow builder (can access other components if needed)

    Yields:
        A configured logging middleware instance
    """
    yield LoggingMiddleware(
        log_level=config.log_level,
        include_inputs=config.include_inputs,
        include_outputs=config.include_outputs
    )
```

### Step 4: Configure in YAML

Add the middleware to your YAML configuration:

```yaml
middleware:
  request_logger:
    _type: logging_middleware
    log_level: DEBUG
    include_inputs: true
    include_outputs: true

functions:
  my_api_function:
    _type: api_call
    endpoint: https://api.example.com
    middleware: ["request_logger"]  # Apply logging middleware
```

### Step 5: Register the Function

Register your function without needing to specify middleware in the decorator:

```python
from nat.cli.register_workflow import register_function
from nat.builder.builder import Builder


@register_function(config_type=MyAPIFunctionConfig)
async def my_api_function(config: MyAPIFunctionConfig, builder: Builder):
    """API function with logging."""
    # Function implementation
    ...
```

## Built-in Middleware

### Cache Middleware

The cache middleware is a built-in component that caches function outputs based on input similarity.

#### Configuration

```yaml
middleware:
  exact_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0  # Exact matching only

  eval_cache:
    _type: cache
    enabled_mode: eval  # Only cache during evaluation
    similarity_threshold: 1.0

  fuzzy_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 0.95  # Allow 95% similarity
```

#### Parameters

- **`enabled_mode`**: `"always"` or `"eval"`
  - `"always"`: Cache is always active
  - `"eval"`: Cache only active when `Context.is_evaluating` is True

- **`similarity_threshold`**: Float from 0.0 to 1.0
  - `1.0`: Exact string matching (fastest)
  - `< 1.0`: Fuzzy matching using `difflib`

#### Usage Example

```yaml
middleware:
  api_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

functions:
  call_external_api:
    _type: api_caller
    endpoint: https://api.example.com
    middleware: ["api_cache"]  # Apply cache middleware
```

```python
@register_function(config_type=APICallerConfig)
async def call_external_api(config: APICallerConfig, builder: Builder):
    """API caller with caching."""
    async def make_api_call(query: str) -> dict:
        # Expensive API call
        response = await external_api.call(query)
        return response

    # Return function implementation
    ...
```

#### Behavior

- **Exact Matching** (threshold=1.0): Uses fast dictionary lookup
- **Fuzzy Matching** (threshold<1.0): Uses `difflib.SequenceMatcher` for similarity
- **Streaming**: Always bypasses cache to avoid buffering
- **Serialization**: Falls back to function call if input can't be serialized

## Advanced Patterns

### Accessing the Builder

Middleware has access to the workflow builder during construction, allowing them to use other components:

```python
@register_middleware(config_type=CachingMiddlewareConfig)
async def caching_middleware(config: CachingMiddlewareConfig, builder: Builder):
    """Middleware that uses an object store for caching."""

    # Access object store component
    object_store = await builder.get_object_store_client(config.object_store_name)

    yield CachingMiddleware(
        object_store=object_store,
        ttl=config.cache_ttl
    )
```

### Final Middleware

Final middleware can short-circuit execution:

```python
class ValidationMiddlewareConfig(FunctionMiddlewareBaseConfig, name="validation"):
    strict_mode: bool = Field(default=True)


class ValidationMiddleware(FunctionMiddleware):
    """Validates inputs and short-circuits on failure."""

    def __init__(self, *, strict_mode: bool):
        super().__init__(is_final=True)  # Mark as final
        self.strict_mode = strict_mode

    async def function_middleware_invoke(self, value, call_next, context):
        # Validate input against schema
        try:
            validated = context.input_schema.model_validate(value)
        except ValidationError as e:
            if self.strict_mode:
                # Short-circuit: don't call next
                raise ValueError(f"Validation failed: {e}")
            else:
                validated = value

        # Only call next if validation passed
        return await call_next(validated)
```

### Chaining Multiple Middleware

Middleware execute in the order specified:

```yaml
middleware:
  logger:
    _type: logging_middleware
    log_level: INFO

  validator:
    _type: validation
    strict_mode: true

  cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

functions:
  protected_function:
    _type: my_function
    middleware: ["logger", "validator", "cache"]  # Execution order
```

```python
@register_function(config_type=MyFunctionConfig)
async def protected_function(config, builder):
    # 1. Logger logs the call
    # 2. Validator validates input
    # 3. Cache checks for cached result or calls function
    ...
```

Execution flow:
```
Request → Logger (pre) → Validator (pre) → Cache (pre) → Function
                                                            ↓
Response ← Logger (post) ← Validator (post) ← Cache (post) ←
```

## Using Middleware with Function Groups

Function groups support middleware at the group level, automatically applying them to all functions in the group. This is useful for applying common middleware (logging, caching, authentication, etc.) across multiple related functions.

### Basic Function Group Middleware

```yaml
middleware:
  api_logger:
    _type: logging_middleware
    log_level: INFO

  api_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

function_groups:
  weather_api:
    _type: weather_api_group
    middleware: ["api_logger", "api_cache"]  # Applied to all functions in the group
```

```python
from nat.cli.register_workflow import register_function_group
from nat.builder.function import FunctionGroup
from nat.data_models.function import FunctionGroupBaseConfig


class WeatherAPIGroupConfig(FunctionGroupBaseConfig, name="weather_api_group"):
    api_key: str


@register_function_group(config_type=WeatherAPIGroupConfig)
async def weather_api_group(config: WeatherAPIGroupConfig, builder):
    """Weather API function group with shared middleware."""
    group = FunctionGroup(config=config)

    async def get_current_weather(location: str) -> dict:
        # All calls to this function will be logged and cached
        return await fetch_weather(location, config.api_key)

    async def get_forecast(location: str, days: int = 5) -> dict:
        # All calls to this function will also be logged and cached
        return await fetch_forecast(location, days, config.api_key)

    group.add_function("get_current_weather", get_current_weather)
    group.add_function("get_forecast", get_forecast)

    yield group
```

### How Function Group Middleware Works

When middleware is configured on a function group:

1. **Automatic Propagation**: All functions added to the group automatically receive the group's middleware
2. **Applied at Creation**: Middleware is configured when each function is added via `add_function()`
3. **Shared Instances**: All functions in the group share the same middleware instances (e.g., shared cache)
4. **Dynamic Updates**: Calling `configure_middleware()` on the group updates all existing functions

### Benefits of Function Group Middleware

**Consistency**: Ensures all related functions have the same middleware
```yaml
function_groups:
  database_operations:
    _type: db_ops_group
    middleware: ["auth_check", "rate_limiter", "query_logger"]
    # All database operations now require auth, are rate-limited, and logged
```

**Maintainability**: Change middleware for all functions in one place
```python
# Dynamically update middleware for all functions in the group
group.configure_middleware([new_logger, new_cache])
```

**Shared State**: Middleware can maintain shared state across all group functions
```yaml
middleware:
  shared_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

function_groups:
  api_group:
    _type: external_api_group
    middleware: ["shared_cache"]
    # Cache is shared across all API functions
```

### Advanced Pattern: Combining Group and Function Middleware

While function groups define middleware at the group level, individual functions can have their own middleware applied after the function is created programmatically if needed. However, the typical pattern is to use group-level middleware for consistency.

## Testing Middleware

### Unit Testing

Test middleware in isolation:

```python
import pytest
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_logging_middleware():
    """Test logging middleware logs correctly."""
    middleware = LoggingMiddleware(
        log_level="DEBUG",
        include_inputs=True,
        include_outputs=True
    )

    # Mock context
    context = FunctionMiddlewareContext(
        name="test_fn",
        config=MagicMock(),
        description="Test",
        input_schema=dict,
        single_output_schema=dict,
        stream_output_schema=None
    )

    # Mock call_next
    async def mock_next(value):
        return {"result": value * 2}

    # Test middleware
    result = await middleware.function_middleware_invoke(5, mock_next, context)
    assert result == {"result": 10}
```

### Integration Testing

Test middleware with actual functions:

```yaml
# test_config.yml
middleware:
  test_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

functions:
  test_function:
    _type: test_func
```

```python
@pytest.mark.asyncio
async def test_function_with_cache():
    """Test function with cache middleware."""
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.data_models.config import Config

    config = Config.from_yaml("test_config.yml")

    async with WorkflowBuilder() as builder:
        workflow = await builder.build_from_config(config)

        # First call
        result1 = await workflow.ainvoke("input")

        # Second call should use cache
        result2 = await workflow.ainvoke("input")

        assert result1 == result2
```

## Best Practices

### Design Principles

1. **Single Responsibility**: Each middleware should do one thing well
2. **Modularity**: Middleware should work well when chained
3. **Configuration**: Make middleware configurable via YAML
4. **Error Handling**: Fail gracefully and log errors
5. **Performance**: Keep middleware lightweight

### Recommended Order

When chaining multiple middleware:

1. **Logging or Monitoring**: First to capture everything
2. **Authentication**: Early rejection of unauthorized calls
3. **Validation**: Validate before expensive operations
4. **Rate Limiting**: Prevent excessive calls
5. **Caching**: Final middleware to skip execution

```yaml
middleware:
  logger:
    _type: logging_middleware
  auth:
    _type: authentication
  validator:
    _type: validation
  rate_limiter:
    _type: rate_limit
  cache:
    _type: cache

functions:
  protected_api:
    _type: api_call
    middleware: ["logger", "auth", "validator", "rate_limiter", "cache"]
```

```python
@register_function(config_type=APIConfig)
async def protected_api(config, builder):
    ...
```

### Build Order

Middleware is built **before** functions and function groups in the workflow builder. This ensures all middleware is available when functions and function groups are constructed.

Build order:
1. Authentication providers
2. Embedders
3. LLMs
4. Memory
5. Object stores
6. Retrievers
7. TTC strategies
8. **Middleware** ← Built here
9. Function groups ← Can use middleware
10. Functions ← Can use middleware

## Troubleshooting

### Common Issues

**Middleware not found error**
```
ValueError: Middleware `my_cache` not found
ValueError: Middleware `my_cache` not found for function group `my_group`
```
Solution: Ensure the middleware is defined in the `middleware` section of your YAML before referencing it in functions or function groups.

**Import errors**
```
ModuleNotFoundError: No module named 'nat.middleware.register'
```
Solution: Ensure the register module is imported. NAT automatically imports `nat.middleware.register` when importing `nat.middleware`.

**Cache not working**
- Check `enabled_mode` setting
- For eval mode, ensure `Context.is_evaluating` is set
- Verify inputs are serializable
- Check similarity threshold

**Performance issues**
- Profile middleware to find bottlenecks
- Use exact matching (threshold=1.0) for caching
- Reduce logging verbosity
- Consider async operations

## API Reference

- {py:class}`~nat.middleware.function_middleware.FunctionMiddleware`: Base class
- {py:class}`~nat.middleware.function_middleware.FunctionMiddlewareContext`: Context info
- {py:class}`~nat.middleware.function_middleware.FunctionMiddlewareChain`: Chain management
- {py:class}`~nat.middleware.register.CacheMiddlewareConfig`: Cache configuration
- {py:class}`~nat.middleware.cache_middleware.CacheMiddleware`: Cache implementation
- {py:func}`~nat.cli.register_workflow.register_middleware`: Registration decorator

## See Also

- [Writing Custom Functions](../extend/functions.md)
- [Function Groups](../extend/function-groups.md)
- [Plugin System](../extend/plugins.md)