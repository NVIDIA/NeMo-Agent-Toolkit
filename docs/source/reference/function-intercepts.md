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

Function intercepts provide a powerful middleware mechanism for adding cross-cutting concerns to functions in the NeMo Agent Toolkit without modifying the function implementation itself. Like middleware in web frameworks (Express.js, FastAPI, etc.), intercepts wrap function calls with a four-phase pattern:

1. **Preprocess** - Inspect and modify inputs before calling next
2. **Call Next** - Delegate to the next middleware or function
3. **Postprocess** - Process, transform, or augment outputs
4. **Continue** - Return or yield the final result

Function intercepts are first-class components in NAT, configured in YAML and built by the workflow builder, just like retrievers, memory providers, and other components.

## Key Concepts

**Function Intercept Component**: A middleware component that:
- Is configured in YAML with a `function_intercepts` section
- Is built by the workflow builder before functions
- Wraps a function's `ainvoke` or `astream` methods
- Can preprocess inputs, postprocess outputs, or short-circuit execution

**Middleware Chain**: A sequence of intercepts that execute in order, forming an "onion" structure where control flows in through preprocessing, down to the function, and back out through postprocessing.

**Final Intercept**: A special middleware marked with `is_final=True` that can terminate the chain. Only one final intercept is allowed per function, and it must be the last in the chain.

## Component-Based Architecture

Function intercepts follow the same component pattern as other NAT components:

```yaml
function_intercepts:
  my_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

  my_logger:
    _type: logging_intercept
    log_level: INFO

functions:
  my_function:
    _type: my_function_type
    # Other function config...
```

```python
@register_function(
    config_type=MyFunctionConfig,
    intercept_names=["my_logger", "my_cache"]  # Reference by name
)
async def my_function(config, builder):
    # Function implementation
    ...
```

## Creating Custom Function Intercepts

### Step 1: Define the Configuration

Create a configuration class inheriting from `FunctionInterceptBaseConfig`:

```python
from pydantic import Field
from nat.data_models.function_intercept import FunctionInterceptBaseConfig


class LoggingInterceptConfig(FunctionInterceptBaseConfig, name="logging_intercept"):
    """Configuration for logging intercept."""

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

### Step 2: Implement the Intercept Class

Create the intercept class inheriting from `FunctionIntercept`:

```python
from nat.intercepts import FunctionIntercept, FunctionInterceptContext
from nat.intercepts import CallNext, CallNextStream
import logging
from typing import Any
from collections.abc import AsyncIterator


class LoggingIntercept(FunctionIntercept):
    """Logging middleware that tracks function calls."""

    def __init__(self, *, log_level: str, include_inputs: bool, include_outputs: bool):
        super().__init__(is_final=False)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.include_inputs = include_inputs
        self.include_outputs = include_outputs

    async def intercept_invoke(
        self,
        value: Any,
        call_next: CallNext,
        context: FunctionInterceptContext
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

    async def intercept_stream(
        self,
        value: Any,
        call_next: CallNextStream,
        context: FunctionInterceptContext
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
# File: src/nat/intercepts/my_module/register.py

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function_intercept
from .logging_intercept import LoggingIntercept, LoggingInterceptConfig


@register_function_intercept(config_type=LoggingInterceptConfig)
async def logging_intercept(config: LoggingInterceptConfig, builder: Builder):
    """Build a logging intercept from configuration.

    Args:
        config: The logging intercept configuration
        builder: The workflow builder (can access other components if needed)

    Yields:
        A configured logging intercept instance
    """
    yield LoggingIntercept(
        log_level=config.log_level,
        include_inputs=config.include_inputs,
        include_outputs=config.include_outputs
    )
```

### Step 4: Configure in YAML

Add the intercept to your YAML configuration:

```yaml
function_intercepts:
  request_logger:
    _type: logging_intercept
    log_level: DEBUG
    include_inputs: true
    include_outputs: true

functions:
  my_api_function:
    _type: api_call
    endpoint: https://api.example.com
```

### Step 5: Reference in Function Registration

Use the intercept by name in your function registration:

```python
from nat.cli.register_workflow import register_function
from nat.builder.builder import Builder


@register_function(
    config_type=MyAPIFunctionConfig,
    intercept_names=["request_logger"]  # Reference by YAML name
)
async def my_api_function(config: MyAPIFunctionConfig, builder: Builder):
    """API function with logging."""
    # Function implementation
    ...
```

## Built-in Intercepts

### Cache Intercept

The cache intercept is a built-in component that memoizes function outputs based on input similarity.

#### Configuration

```yaml
function_intercepts:
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

- **enabled_mode**: `"always"` or `"eval"`
  - `"always"`: Cache is always active
  - `"eval"`: Cache only active when `Context.is_evaluating` is True

- **similarity_threshold**: Float from 0.0 to 1.0
  - `1.0`: Exact string matching (fastest)
  - `< 1.0`: Fuzzy matching using difflib

#### Usage Example

```yaml
function_intercepts:
  api_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0

functions:
  call_external_api:
    _type: api_caller
    endpoint: https://api.example.com
```

```python
@register_function(
    config_type=APICallerConfig,
    intercept_names=["api_cache"]
)
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
- **Fuzzy Matching** (threshold<1.0): Uses difflib.SequenceMatcher for similarity
- **Streaming**: Always bypasses cache to avoid buffering
- **Serialization**: Falls back to function call if input can't be serialized

## Advanced Patterns

### Accessing the Builder

Intercepts have access to the workflow builder during construction, allowing them to use other components:

```python
@register_function_intercept(config_type=CachingInterceptConfig)
async def caching_intercept(config: CachingInterceptConfig, builder: Builder):
    """Intercept that uses an object store for caching."""

    # Access object store component
    object_store = await builder.get_object_store_client(config.object_store_name)

    yield CachingIntercept(
        object_store=object_store,
        ttl=config.cache_ttl
    )
```

### Final Intercepts

Final intercepts can short-circuit execution:

```python
class ValidationInterceptConfig(FunctionInterceptBaseConfig, name="validation"):
    strict_mode: bool = Field(default=True)


class ValidationIntercept(FunctionIntercept):
    """Validates inputs and short-circuits on failure."""

    def __init__(self, *, strict_mode: bool):
        super().__init__(is_final=True)  # Mark as final
        self.strict_mode = strict_mode

    async def intercept_invoke(self, value, call_next, context):
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

### Chaining Multiple Intercepts

Intercepts execute in the order specified:

```yaml
function_intercepts:
  logger:
    _type: logging_intercept
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
```

```python
@register_function(
    config_type=MyFunctionConfig,
    intercept_names=["logger", "validator", "cache"]  # Execution order
)
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

## Testing Intercepts

### Unit Testing

Test intercepts in isolation:

```python
import pytest
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_logging_intercept():
    """Test logging intercept logs correctly."""
    intercept = LoggingIntercept(
        log_level="DEBUG",
        include_inputs=True,
        include_outputs=True
    )

    # Mock context
    context = FunctionInterceptContext(
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

    # Test intercept
    result = await intercept.intercept_invoke(5, mock_next, context)
    assert result == {"result": 10}
```

### Integration Testing

Test intercepts with actual functions:

```yaml
# test_config.yml
function_intercepts:
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
    """Test function with cache intercept."""
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

1. **Single Responsibility**: Each intercept should do one thing well
2. **Composability**: Intercepts should work well when chained
3. **Configuration**: Make intercepts configurable via YAML
4. **Error Handling**: Fail gracefully and log errors
5. **Performance**: Keep intercepts lightweight

### Recommended Order

When chaining multiple intercepts:

1. **Logging/Monitoring**: First to capture everything
2. **Authentication**: Early rejection of unauthorized calls
3. **Validation**: Validate before expensive operations
4. **Rate Limiting**: Prevent excessive calls
5. **Caching**: Final intercept to skip execution

```yaml
function_intercepts:
  logger:
    _type: logging_intercept
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
```

```python
@register_function(
    config_type=APIConfig,
    intercept_names=["logger", "auth", "validator", "rate_limiter", "cache"]
)
async def protected_api(config, builder):
    ...
```

### Build Order

Function intercepts are built **before** functions in the workflow builder. This ensures all intercepts are available when functions are constructed.

Build order:
1. Authentication providers
2. Embedders
3. LLMs
4. Memory
5. Object stores
6. Retrievers
7. TTC strategies
8. **Function intercepts** ← Built here
9. Function groups
10. Functions ← Use intercepts here

## Troubleshooting

### Common Issues

**Intercept not found error**
```
ValueError: Function intercept `my_cache` not found
```
Solution: Ensure the intercept is defined in the `function_intercepts` section of your YAML.

**Import errors**
```
ModuleNotFoundError: No module named 'nat.intercepts.register'
```
Solution: Ensure the register module is imported. NAT automatically imports `nat.intercepts.register` when importing `nat.intercepts`.

**Cache not working**
- Check `enabled_mode` setting
- For eval mode, ensure `Context.is_evaluating` is set
- Verify inputs are serializable
- Check similarity threshold

**Performance issues**
- Profile intercepts to find bottlenecks
- Use exact matching (threshold=1.0) for caching
- Reduce logging verbosity
- Consider async operations

## Migration from Old Pattern

If you have existing code using the old pattern:

**Old Pattern** (instantiate at registration):
```python
@register_function(
    config_type=MyFunctionConfig,
    intercepts=[CacheIntercept(enabled_mode="always")]
)
async def my_function(config, builder):
    ...
```

**New Pattern** (reference by name):
```yaml
function_intercepts:
  my_cache:
    _type: cache
    enabled_mode: always
    similarity_threshold: 1.0
```

```python
@register_function(
    config_type=MyFunctionConfig,
    intercept_names=["my_cache"]
)
async def my_function(config, builder):
    ...
```

## API Reference

- {py:class}`~nat.intercepts.function_intercept.FunctionIntercept`: Base class
- {py:class}`~nat.intercepts.function_intercept.FunctionInterceptContext`: Context info
- {py:class}`~nat.intercepts.function_intercept.FunctionInterceptChain`: Chain management
- {py:class}`~nat.intercepts.register.CacheInterceptConfig`: Cache configuration
- {py:class}`~nat.intercepts.cache_intercept.CacheIntercept`: Cache implementation
- {py:func}`~nat.cli.register_workflow.register_function_intercept`: Registration decorator

## See Also

- [Writing Custom Functions](../extend/functions.md)
- [Function Groups](../extend/function-groups.md)
- [Plugin System](../extend/plugins.md)
