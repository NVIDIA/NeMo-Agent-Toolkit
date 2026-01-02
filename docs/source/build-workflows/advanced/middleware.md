<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

Middleware provides a powerful mechanism for adding cross-cutting concerns to functions in the NeMo Agent toolkit without modifying the function implementation itself. Like middleware in web frameworks (Express.js, FastAPI, etc.), middleware wraps function calls with a four-phase pattern:

1. **Preprocess** - Inspect and modify inputs before calling next
2. **Call Next** - Delegate to the next middleware or function
3. **Postprocess** - Process, transform, or augment outputs
4. **Continue** - Return or yield the final result

Middleware components are first-class components in NeMo Agent toolkit, configured in YAML and built by the workflow builder, just like retrievers, [memory](../memory.md) providers, and other components.

## Key Concepts

**Middleware Component**: A middleware component that:
- Is configured in YAML with a `middleware` section
- Is built by the workflow builder before [functions](../functions-and-function-groups/functions.md) and [function groups](../functions-and-function-groups/function-groups.md)
- Wraps a function's `ainvoke` or `astream` methods
- Can be applied to individual functions or entire function groups
- Can preprocess inputs, postprocess outputs, or short-circuit execution

**Middleware Chain**: A sequence of middleware that execute in order, forming an "onion" structure where control flows in through preprocessing, down to the function, and back out through postprocessing.

**Final Middleware**: A special middleware marked with `is_final=True` that can terminate the chain. Only one final middleware is allowed per function, and it must be the last in the chain.

## Component-Based Architecture

Middleware follows the same component pattern as other components:

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

Create a configuration class inheriting from `DynamicMiddlewareConfig`:

```python
from pydantic import Field
from nat.middleware.dynamic.dynamic_middleware_config import DynamicMiddlewareConfig


class LoggingMiddlewareConfig(DynamicMiddlewareConfig, name="logging_middleware"):
    """Configuration for logging middleware.

    Inherits dynamic discovery features (register_llms, register_workflow_functions,
    and so on) and the enabled toggle from DynamicMiddlewareConfig.
    """

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
```

The `DynamicMiddlewareConfig` base class provides the following fields:

**Enable/Disable:**

- `enabled` (`bool`, default=`True`): Toggle middleware on or off at runtime through configuration

**Auto-Discovery Flags:**

When set to `True`, these flags automatically intercept all components of that type:

- `register_llms` (`bool`, default=`False`): Auto-discover and intercept all LLM component functions
- `register_embedders` (`bool`, default=`False`): Auto-discover and intercept all embedder component functions
- `register_retrievers` (`bool`, default=`False`): Auto-discover and intercept all retriever component functions
- `register_memory` (`bool`, default=`False`): Auto-discover and intercept all memory provider component functions
- `register_object_stores` (`bool`, default=`False`): Auto-discover and intercept all object store component functions
- `register_auth_providers` (`bool`, default=`False`): Auto-discover and intercept all authentication provider component functions
- `register_workflow_functions` (`bool`, default=`False`): Auto-discover and intercept all workflow functions

**Explicit Component References:**

For fine-grained control, specify exactly which components to intercept (alternative to auto-discovery):

- `llms` (list, default=`[]`): Specific LLM component names to intercept
- `embedders` (list, default=`[]`): Specific embedder component names to intercept
- `retrievers` (list, default=`[]`): Specific retriever component names to intercept
- `memory` (list, default=`[]`): Specific memory provider component names to intercept
- `object_stores` (list, default=`[]`): Specific object store component names to intercept
- `auth_providers` (list, default=`[]`): Specific authentication provider component names to intercept

**Function Allow Lists:**

- `allowed_component_functions` (object, default=`None`): Controls which methods on each component type can be wrapped. When `None`, uses built-in defaults. Provide to extend the defaults with additional method names:
  - `llms` (set of strings): Additional LLM methods to allow
  - `embedders` (set of strings): Additional embedder methods to allow
  - `retrievers` (set of strings): Additional retriever methods to allow
  - `memory` (set of strings): Additional memory methods to allow
  - `object_stores` (set of strings): Additional object store methods to allow
  - `authentication` (set of strings): Additional authentication methods to allow

**How toggles and allow lists interact:**

1. Auto-discovery flags (`register_*`) control *which components* are intercepted
2. Explicit references (`llms`, `embedders`, and so on) provide fine-grained component selection
3. `allowed_component_functions` controls *which methods* on those components can be wrapped
4. Only methods in the allowlist are wrapped; others pass through unchanged

**Default Allowed Functions by Component Type:**

The following methods are allowed by default for each component type. You can extend these lists through `allowed_component_functions`:

| Component Type | Default Allowed Methods |
|----------------|------------------------|
| **LLMs** | `invoke`, `ainvoke`, `stream`, `astream` |
| **Embedders** | `embed_query`, `aembed_query` |
| **Retrievers** | `search` |
| **Memory** | `search`, `add_items`, `remove_items` |
| **Object Stores** | `put_object`, `get_object`, `delete_object`, `upsert_object` |
| **Authentication** | `authenticate` |

Workflow functions (`register_workflow_functions`) intercept the function's `ainvoke` and `astream` methods directly.

### Step 2: Implement the Middleware Class

Create the middleware class inheriting from `DynamicFunctionMiddleware`:

```python
import logging

from nat.middleware.dynamic.dynamic_function_middleware import DynamicFunctionMiddleware
from nat.middleware.middleware import InvocationContext

logger = logging.getLogger(__name__)


class LoggingMiddleware(DynamicFunctionMiddleware):
    """Logging middleware that tracks function calls.

    Extends DynamicFunctionMiddleware to get automatic chain orchestration
    and dynamic discovery features. Custom logic is implemented through
    the pre_invoke and post_invoke hooks.
    """

    async def pre_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Log inputs before function execution.

        Args:
            context: Invocation context containing:
                - function_context: Static function metadata (frozen)
                - original_args: Original function arguments before transformation (frozen)
                - original_kwargs: Original function keyword arguments before transformation (frozen)
                - modified_args: Current function arguments (mutable)
                - modified_kwargs: Current function keyword arguments (mutable)
                - output: None (function not yet called)

        Returns:
            InvocationContext if modified, or None to pass through unchanged
        """
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        logger.log(log_level, f"Calling {context.function_context.name} with args: {context.modified_args}")

        # Optional: Check if args were modified by prior middleware
        if context.modified_args != context.original_args:
            logger.log(log_level, f"  (original args were: {context.original_args})")

        return None  # Pass through unchanged

    async def post_invoke(self, context: InvocationContext) -> InvocationContext | None:
        """Log outputs after function execution.

        Args:
            context: Invocation context (Pydantic model) containing:
                - function_context: Static function metadata (frozen)
                - original_args: Original function arguments before transformation (frozen)
                - original_kwargs: Original function keyword arguments before transformation (frozen)
                - modified_args: Function arguments after pre-invoke transforms (mutable)
                - modified_kwargs: Function keyword arguments after pre-invoke transforms (mutable)
                - output: Current output value (mutable)

        Returns:
            InvocationContext if modified, or None to pass through unchanged
        """
        log_level = getattr(logging, self._config.log_level.upper(), logging.INFO)
        logger.log(log_level, f"Function {context.function_context.name} returned: {context.output}")
        return None  # Pass through unchanged
```

Key benefits of extending `DynamicFunctionMiddleware`:

- **No manual chain handling**: The base class manages `call_next` orchestration automatically
- **Separate hooks**: `pre_invoke` handles input processing, `post_invoke` handles output processing
- **Unified context**: Single `InvocationContext` used for both phases
  - Pre-invoke: `output` is `None`, modify `modified_args`/`modified_kwargs`
  - Post-invoke: `output` has the result, modify to transform
- **Chain awareness**: Access `original_args` to see original values versus current `modified_args`
- **Frozen originals**: `original_args`/`original_kwargs` are immutable (Pydantic enforced)
- **Mutable current values**: Modify `modified_args`/`modified_kwargs`/`output` in place, return context to signal changes
- **Streaming support built-in**: `post_invoke` is called per-chunk for streaming functions
- **Configuration access**: Use `self._config` to access your configuration values

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
    yield LoggingMiddleware(config=config, builder=builder)
```

### Step 4: Configure in YAML

Add the middleware to your YAML configuration:

```yaml
middleware:
  request_logger:
    _type: logging_middleware
    log_level: DEBUG
    enabled: true  # Inherited from DynamicMiddlewareConfig
    # Dynamic discovery options (inherited):
    # register_llms: true
    # register_workflow_functions: true

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

    async def function_middleware_invoke(self, *args, call_next, context, **kwargs):
        # Validate input against schema (using first arg)
        value = args[0] if args else None
        try:
            validated = context.input_schema.model_validate(value)
        except ValidationError as e:
            if self.strict_mode:
                # Short-circuit: don't call next
                raise ValueError(f"Validation failed: {e}")
            else:
                validated = value

        # Only call next if validation passed
        return await call_next(validated, *args[1:], **kwargs)
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
from nat.middleware.middleware import FunctionMiddlewareContext, InvocationContext


@pytest.mark.asyncio
async def test_logging_middleware():
    """Test logging middleware logs correctly."""
    # Create a mock config
    mock_config = MagicMock()
    mock_config.log_level = "DEBUG"
    mock_config.enabled = True

    # Create a mock builder
    mock_builder = MagicMock()

    # Create middleware instance
    middleware = LoggingMiddleware(config=mock_config, builder=mock_builder)

    # Mock function context (static metadata only - no args/kwargs)
    function_context = FunctionMiddlewareContext(
        name="test_fn",
        config=MagicMock(),
        description="Test",
        input_schema=dict,
        single_output_schema=dict,
        stream_output_schema=None
    )

    # Test pre_invoke (output is None, function not yet called)
    context = InvocationContext(
        function_context=function_context,
        original_args=(5,),        # Frozen - original function args
        original_kwargs={},        # Frozen - original function kwargs
        modified_args=(5,),        # Mutable - current args
        modified_kwargs={},        # Mutable - current kwargs
        output=None                # None in pre-invoke phase
    )
    result = await middleware.pre_invoke(context)
    assert result is None  # Pass-through, no modification

    # Test post_invoke (output now has the result)
    context.output = {"result": 10}  # Set output after function call
    result = await middleware.post_invoke(context)
    assert result is None  # Pass-through, no modification

    # Test detecting modified args
    context_modified = InvocationContext(
        function_context=function_context,
        original_args=(5,),        # Original
        original_kwargs={},
        modified_args=(10,),       # Modified - different from original_args
        modified_kwargs={},
        output=None
    )
    # Middleware can detect: context_modified.modified_args != context_modified.original_args
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
1. [Authentication providers](../../components/auth/api-authentication.md)
2. [Embedders](../embedders.md)
3. [LLMs](../llms/index.md)
4. [Memory](../memory.md)
5. [Object stores](../object-store.md)
6. [Retrievers](../retrievers.md)
7. [TTC strategies](../../improve-workflows/test-time-compute.md)
8. **Middleware** ← Built here
9. [Function groups](../functions-and-function-groups/function-groups.md) ← Can use middleware
10. [Functions](../functions-and-function-groups/functions.md) ← Can use middleware

## Dynamic Middleware: Unregistering Callables

The `DynamicFunctionMiddleware` supports unregistering callables at runtime, allowing you to remove middleware interception from workflow functions or component methods.

### Unregister API

The `unregister` method accepts a `RegisteredFunction` or `RegisteredComponentMethod` object. Use the `get_registered()` method to retrieve a registered callable by its key:

```python
from nat.middleware.utils.workflow_inventory import RegisteredFunction, RegisteredComponentMethod

# Get a registered callable by key
registered = middleware.get_registered("my_llm.invoke")

# Unregister it (if found)
if registered:
    middleware.unregister(registered)

# List all registered keys
all_keys = middleware.get_registered_keys()
```

### Behavior

- **Workflow Functions**: Removes the `DynamicFunctionMiddleware` from the function's middleware chain
- **Component Methods**: Restores the original unwrapped method on the component instance

### Registered Callable Models

The tracking uses Pydantic models for type safety:

- **`RegisteredFunction`**: Tracks workflow functions with `key` and `function_instance`
- **`RegisteredComponentMethod`**: Tracks component methods with `key`, `component_instance`, `function_name`, and `original_callable`
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
Solution: Ensure the register module is imported. NeMo Agent toolkit automatically imports `nat.middleware.register` when importing `nat.middleware`.

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
- {py:class}`~nat.middleware.cache.cache_middleware_config.CacheMiddlewareConfig`: Cache configuration
- {py:class}`~nat.middleware.cache.cache_middleware.CacheMiddleware`: Cache implementation
- {py:func}`~nat.cli.register_workflow.register_middleware`: Registration decorator

## See Also

- [Writing Custom Functions](../../extend/custom-components/custom-functions/functions.md)
- [Function Groups](../../extend/custom-components/custom-functions/function-groups.md)
- [Plugin System](../../extend/plugins.md)
