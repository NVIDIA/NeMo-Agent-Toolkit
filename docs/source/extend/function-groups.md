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

Function groups bundle related functions that share configuration and runtime context.
Use them to centralize resource management (for example, database connections) and to group functions by namespace.
It is also possible to selectively expose functions, enabling namespace isolation of functions within the group and making them addressable as ordinary functions.

## Define the Configuration

Create a configuration class that inherits from {py:class}`~nat.data_models.function.FunctionGroupBaseConfig`. Use Pydantic fields for validation and documentation.

The optional `include` list controls which functions in the group become globally addressable and accessible to the workflow builder.
The optional `exclude` list controls which functions in the group should not be addressable and accessible to the workflow builder and also not be wrapped as tools.

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

- Functions are referenced as `instance_name.function_name` (for example, `my.greet`).
- Only functions listed in `config.include` are added to the global registry.
- If both `include` and `exclude` are empty, no functions are globally added, but you can still access the group and its functions programmatically.
- If `exclude` is provided, matching functions are filtered out from default exposure, but they remain accessible programmatically via the group.

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
    accessible_functions = my_group.get_accessible_functions()
    
    # Get all functions in the group.
    # This will return all functions in the group, regardless of whether they are included or excluded.
    all_functions = my_group.get_all_functions()

    # Or only the included functions (which have also been registered globally as ordinary functions)
    included_functions = my_group.get_included_functions()

    # Or only the excluded functions
    excluded_functions = my_group.get_excluded_functions()
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

## Best Practices

- Keep group instance names short and descriptive; they become part of function names.
- Share expensive resources (for example, clients, caches) through the group context instead of recreating them per function.
- Validate inputs with Pydantic schemas for robust error handling.

## Next Steps

- Review [Writing Custom Functions](./functions.md) for details that also apply to functions inside groups (type safety, streaming vs. single outputs, converters).
