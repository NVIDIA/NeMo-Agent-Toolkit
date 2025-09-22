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

# Writing Custom Function Groups

It is strongly recommended to first read the [Function Groups](../workflows/function-groups.md) guide before reading this guide.

Function groups bundle related functions that share configuration and runtime context.
Use them to centralize resource management (for example, database connections) and to group functions by namespace.
It is also possible to selectively include and exclude functions, from the function group. Selective inclusion enables namespace isolation of functions within the group and makes them addressable as ordinary functions.

## Define the Configuration

Create a configuration class that inherits from {py:class}`~nat.data_models.function.FunctionGroupBaseConfig`. Use Pydantic fields for validation and documentation.

The optional `include` list controls which functions in the group become globally addressable and accessible to the workflow builder. If any function from the group is not listed in `include`, it will not be added to the global registry.
The optional `exclude` list controls which functions in the group should not be addressable and accessible to the workflow builder and also not be wrapped as tools.
If both `include` and `exclude` are empty, no functions are globally added, but you can still access the group and its functions programmatically.

:::{note}
`include` and `exclude` are mutually exclusive. If both are provided, a `ValueError` will be raised.
:::

```python
from pydantic import Field
from nat.data_models.function import FunctionGroupBaseConfig


class MyGroupConfig(FunctionGroupBaseConfig, name="my_group"):
    pass
```

## Register the Function Group

Register using {py:func}`nat.cli.register_workflow.register_function_group`. The registered coroutine should yield a {py:class}`~nat.builder.function.FunctionGroup` instance.

```python
from nat.cli.register_workflow import register_function_group
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup

@register_function_group(config_type=MyGroupConfig)
async def build_my_group(config: MyGroupConfig, _builder: Builder):
    group = FunctionGroup(config=config, instance_name="my")

    async def greet_fn(name: str) -> str:
        """Return a friendly greeting given a name."""
        return f"Hello, {name}!"

    async def shout_fn(message: str) -> str:
        """Return a message in uppercase."""
        return message.upper()

    group.add_function(name="greet", fn=greet_fn, description=greet_fn.__doc__)
    group.add_function(name="shout", fn=shout_fn, description=shout_fn.__doc__)

    yield group
```

## Referencing Functions within a Function Group

Functions are referenced as `instance_name.function_name` (for example, `my.greet`).

```python
async with WorkflowBuilder() as builder:
    await builder.add_function_group("my", MyGroupConfig(include=["greet"]))

    # Able to reference the function directly by its fully qualified name
    greet = builder.get_function("my.greet")
    print(await greet.ainvoke("World"))

    my_group = builder.get_function_group("my")

    # Get all accessible functions in the function group.
    # If the function group is configured to:
    # - include some functions, this will return only the included functions.
    # - not include or exclude any function, this will return all functions in the group.
    # - exclude some functions, this will return all functions in the group except the excluded functions.
    accessible_functions = await my_group.get_accessible_functions()
    
    # Get all functions in the group.
    # This will return all functions in the group, regardless of whether they are included or excluded.
    all_functions = await my_group.get_all_functions()

    # Or only the included functions (which have also been registered globally as ordinary functions)
    included_functions = await my_group.get_included_functions()

    # Or only the excluded functions
    excluded_functions = await my_group.get_excluded_functions()
```

## Input Schemas

When you add a function with {py:meth}`~nat.builder.function.FunctionGroup.add_function`, the toolkit infers input/output schemas from the callable's type hints. You can override the input schema and attach converters if needed:

```python
from pydantic import BaseModel, Field
from nat.builder.function_info import FunctionInfo


class GreetingInput(BaseModel):
    name: str = Field(description="Name to greet", min_length=1)


async def greet_fn(name: str) -> str:
    return f"Hello, {name}!"


group.add_function(name="greet",
                   fn=greet_fn,
                   input_schema=GreetingInput,
                   description="Return a friendly greeting")
```

## Using Filters

Function groups support dynamic filtering to control which functions are accessible at runtime. Filters are applied when functions are accessed, not when they are added to the group.

### Group-Level Filters

Group-level filters receive a list of function names and return a filtered list. Set them during group creation or use {py:meth}`~nat.builder.function.FunctionGroup.set_filter_fn`:

```python
from collections.abc import Sequence

@register_function_group(config_type=MyGroupConfig)
async def build_my_group(config: MyGroupConfig, _builder: Builder):
    # Define a group-level filter
    async def admin_filter(function_names: Sequence[str]) -> Sequence[str]:
        # Only include admin functions in production
        if config.environment == "production":
            return [name for name in function_names if name.startswith("admin_")]
        return function_names
    
    # Create group with filter
    group = FunctionGroup(config=config, instance_name="my", filter_fn=admin_filter)
    
    # Or set filter later
    # group.set_filter_fn(admin_filter)
    
    # Add functions as normal
    group.add_function("admin_reset", reset_fn)
    group.add_function("user_greet", greet_fn)
    
    yield group
```

### Per-Function Filters

Per-function filters are set on individual functions and receive the function name. They determine whether that specific function should be included:

```python
@register_function_group(config_type=MyGroupConfig)
async def build_my_group(config: MyGroupConfig, _builder: Builder):
    group = FunctionGroup(config=config, instance_name="my")
    
    # Define per-function filter
    async def debug_only(name: str) -> bool:
        return config.debug_mode  # Only include if debug mode is enabled
    
    async def debug_fn(message: str) -> str:
        return f"DEBUG: {message}"
    
    # Add function with per-function filter
    group.add_function(name="debug", 
                       fn=debug_fn, 
                       filter_fn=debug_only)
    
    # Or set filter after adding
    # group.set_per_function_filter_fn("debug", debug_only)
    
    yield group
```

### Filter Interaction

Filters work in combination with `include` and `exclude` configuration:

1. **Configuration filtering**: Applied first based on `include`/`exclude` lists
2. **Group-level filtering**: Applied to the result of configuration filtering  
3. **Per-function filtering**: Applied to each remaining function individually

```python
class FilteredGroupConfig(FunctionGroupBaseConfig, name="filtered_group"):
    include: list[str] = ["func1", "func2", "func3"]
    environment: str = "development"

@register_function_group(config_type=FilteredGroupConfig)
async def build_filtered_group(config: FilteredGroupConfig, _builder: Builder):
    # Group filter: only production-ready functions in production
    async def env_filter(names: Sequence[str]) -> Sequence[str]:
        if config.environment == "production":
            return [name for name in names if not name.startswith("test_")]
        return names
    
    # Per-function filter: exclude experimental features
    async def stable_only(name: str) -> bool:
        return not name.endswith("_experimental")
    
    group = FunctionGroup(config=config, filter_fn=env_filter)
    
    group.add_function("func1", fn1)  # Included by config
    group.add_function("test_func2", fn2)  # Included by config, but filtered by env_filter in production
    group.add_function("func3_experimental", fn3, filter_fn=stable_only)  # Excluded by per-function filter
    
    yield group
```

## Best Practices

- Keep group instance names short and descriptive; they become part of function names.
- Share expensive resources (for example, clients, caches) through the group context instead of recreating them per function.
- Validate inputs with Pydantic schemas for robust error handling.
- Use group-level filters for environment-based or broad categorical filtering (for example, production readiness, feature flags).
- Use per-function filters for function-specific conditions (for example, user permissions, feature availability).
- Apply filters that depend on runtime configuration rather than static conditions that could be handled by `include`/`exclude` lists.
- Test filter logic thoroughly, as filters are applied dynamically and can change function availability based on runtime conditions.

## Next Steps

- Review [Writing Custom Functions](./functions.md) for details that also apply to functions inside groups (type safety, streaming vs. single outputs, converters).
