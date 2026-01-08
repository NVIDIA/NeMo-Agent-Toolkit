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

# Writing Per-User Functions

Per-user functions provide user-isolated state for multi-user deployments. Unlike shared functions that are built once at startup and shared across all users, per-user functions are instantiated lazily when a user's first request arrives, with each user receiving their own isolated instance.

:::{note}
**If your workflow uses per-user functions or function groups, the workflow itself must also be registered as per-user.** Shared workflows cannot access per-user components. See [Dependency Rules](#dependency-rules) for details.
:::

## When to Use Per-User Functions

Per-user functions are useful when you need:

- **User-isolated state**: Each user's data is completely separate from other users
- **Stateful functions**: Functions that maintain state across requests for the same user
- **Session-based personalization**: User preferences, history, or context that persists within a session
- **Per-user resources**: Resources like database connections or MCP clients that should be isolated per user

## Registering Per-User Functions

### The `@register_per_user_function` Decorator

To register a per-user function, use the {py:deco}`nat.cli.register_workflow.register_per_user_function` decorator. This decorator is similar to {py:deco}`nat.cli.register_workflow.register_function` but requires explicit schema definitions for input and output types.

```python
from pydantic import BaseModel, Field
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_per_user_function
from nat.data_models.function import FunctionBaseConfig


# Define input and output schemas
class MyInput(BaseModel):
    message: str = Field(description="Input message")


class MyOutput(BaseModel):
    response: str = Field(description="Response message")
    count: int = Field(description="Number of times this user called the function")


# Define the configuration
class MyPerUserFunctionConfig(FunctionBaseConfig, name="my_per_user_function"):
    greeting: str = Field(default="Hello", description="Greeting to use")


# Register the per-user function
@register_per_user_function(
    config_type=MyPerUserFunctionConfig,
    input_type=MyInput,
    single_output_type=MyOutput
)
async def my_per_user_function(config: MyPerUserFunctionConfig, builder: Builder):
    # This state is unique per user - created fresh for each user
    user_counter = {"count": 0}

    async def _impl(inp: MyInput) -> MyOutput:
        user_counter["count"] += 1
        return MyOutput(
            response=f"{config.greeting}, {inp.message}!",
            count=user_counter["count"]
        )

    yield FunctionInfo.from_fn(_impl)
```

### Required Schema Parameters

Unlike the regular `@register_function` decorator, `@register_per_user_function` requires explicit schema definitions because per-user functions are instantiated lazily. The schemas enable documentation generation without requiring a concrete function instance at startup.


### Schema Type Flexibility

The `input_type`, `single_output_type`, and `streaming_output_type` parameters accept **both Pydantic models and simple Python types**. Simple types are automatically converted to Pydantic schemas.

```python
# Using Pydantic models (recommended for complex types)
@register_per_user_function(
    config_type=MyConfig,
    input_type=MyInputModel,
    single_output_type=MyOutputModel
)
async def with_pydantic_schemas(config, builder):
    ...

# Using simple Python types
@register_per_user_function(
    config_type=MyConfig,
    input_type=str,
    single_output_type=int
)
async def with_simple_types(config, builder):
    async def _impl(message: str) -> int:
        return len(message)
    yield FunctionInfo.from_fn(_impl)
```

## Registering Per-User Function Groups

Function groups that need per-user state can be registered using the {py:deco}`nat.cli.register_workflow.register_per_user_function_group` decorator.

```python
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.function import FunctionGroupBaseConfig
from nat.builder.function import FunctionGroup


class MyPerUserGroupConfig(FunctionGroupBaseConfig, name="my_per_user_group"):
    setting: str = Field(default="default", description="Group-wide setting")


@register_per_user_function_group(config_type=MyPerUserGroupConfig)
async def my_per_user_group(config: MyPerUserGroupConfig, builder: Builder):
    # Per-user state shared across all functions in this group
    group_state = {"calls": 0}

    class MyGroup(FunctionGroup):
        def get_included_functions(self):
            # Return functions that share this per-user state
            ...

    yield MyGroup()
```

## How Per-User Functions Work

### User Identification

When using the FastAPI front end with `nat serve`, users are identified by the `nat-session` cookie. Each unique session ID represents a different user.

```bash
# User "alice" makes a request
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d ''{"messages": [{"role": "user", "content": "Hello"}]}''

# User "bob" makes a request (isolated from alice)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d ''{"messages": [{"role": "user", "content": "Hello"}]}''
```

### Lifecycle

1. **Lazy instantiation**: Per-user functions are not built at server startup. Instead, they are created when a user's first request arrives.

2. **State isolation**: Each user gets their own instance of the function with separate state. State changes by one user do not affect other users.

3. **Automatic cleanup**: Inactive user sessions are automatically cleaned up based on the configured timeout (`per_user_workflow_timeout` in the configuration).

### Configuration Options

The following configuration options control per-user function behavior:

| Option | Default | Description |
|--------|---------|-------------|
| `per_user_workflow_timeout` | 30 minutes | How long inactive user sessions are kept |
| `per_user_workflow_cleanup_interval` | 5 minutes | How often to check for inactive sessions |
| `enable_per_user_monitoring` | false | Enable the `/monitor/users` endpoint for resource monitoring |

### Monitoring Per-User Workflows

The NVIDIA NeMo Agent toolkit provides a built-in monitoring endpoint for per-user workflows that exposes real-time resource usage metrics. To enable it, set `enable_per_user_monitoring` to `true` in your configuration:

```yaml
general:
  enable_per_user_monitoring: true
```

The `/monitor/users` endpoint provides metrics including:

- **Session lifecycle**: Creation time, last activity, active status
- **Request metrics**: Total requests, active requests, average latency, error count
- **LLM usage**: Token counts (prompt, completion, total), LLM API calls
- **Resource counts**: Number of per-user functions and function groups built

For complete API documentation and usage examples, refer to [Per-User Workflow Monitoring Endpoint](../../../reference/rest-api/api-server-endpoints.md#per-user-workflow-monitoring-endpoint).

## Dependency Rules

Per-user functions have specific dependency rules that ensure proper isolation.

:::{important}
**To use per-user functions or function groups, the workflow itself must be registered as per-user using `@register_per_user_function`.**

A shared workflow (registered with `@register_function`) cannot access per-user functions because shared workflows are built once at startup before any user context exists. If your workflow needs to use per-user functions, you must register the workflow with `@register_per_user_function`.
:::

### Allowed Dependencies

- **Per-user workflows can depend on per-user functions**: A per-user workflow can call `builder.get_function()` to access per-user functions. Each user gets their own isolated instances.

- **Per-user functions can depend on shared functions**: A per-user function can call `builder.get_function()` to access shared functions. The shared function instance is the same for all users.

- **Per-user functions can depend on other per-user functions**: The dependency will be resolved within the same user's builder, ensuring proper isolation.

### Prohibited Dependencies

- **Shared workflows cannot depend on per-user functions**: A shared workflow cannot call `builder.get_function()` on a per-user function. This restriction exists because shared workflows are built at startup before any user requests arrive, while per-user functions require a user context.

- **Shared functions cannot depend on per-user functions**: Similarly, a shared function cannot access per-user functions.

```python
# This will raise an error during startup
@register_function(config_type=SharedFunctionConfig)
async def shared_function(config, builder):
    # ERROR: Cannot access per-user function from shared function
    per_user_fn = await builder.get_function("my_per_user_function")
    ...
```

## Complete Example

For a complete working example demonstrating per-user functions and workflows, see the example in `examples/front_ends/per_user_workflow`.

This example includes:
- Per-user notepad function with isolated note storage
- Per-user preferences function with user-specific settings
- Per-user workflow that orchestrates multiple per-user functions
- Session statistics tracking per user
- Complete configuration and usage instructions

## Best Practices

1. **Define clear schemas**: Always provide well-documented Pydantic models for input and output types to ensure good API documentation.

2. **Consider memory usage**: Each user gets their own function instance, so be mindful of memory usage in state variables, especially for high-traffic applications.

3. **Use appropriate timeouts**: Configure `per_user_workflow_timeout` based on your use case. Shorter timeouts reduce memory usage but may cause more frequent re-initialization.

4. **Handle cleanup gracefully**: If your per-user function uses external resources (database connections, file handles, and so on), use the async context manager pattern to ensure proper cleanup:

    ```python
    @register_per_user_function(
        config_type=MyConfig,
        input_type=MyInput,
        single_output_type=MyOutput
    )
    async def my_function(config, builder):
        # Initialization
        connection = await create_connection()

        try:
            async def _impl(inp):
                # Use connection
                ...

            yield FunctionInfo.from_fn(_impl)
        finally:
            # Cleanup when user session ends
            await connection.close()
    ```

5. **Validate dependencies at development time**: Test your workflows to ensure you're not accidentally creating dependencies from shared functions to per-user functions.
