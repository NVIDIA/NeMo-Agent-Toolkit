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

Function groups bundle related functions that share configuration and runtime context. Use them to centralize resource management (for example, database connections), group functions by namespace, and selectively expose functions globally.

## Define the Configuration

Create a configuration class that inherits from {py:class}`~nat.data_models.function.FunctionGroupBaseConfig`. Use Pydantic fields for validation and documentation. The optional `expose` list controls which functions in the group become globally addressable.

```python
from pydantic import Field
from nat.data_models.function import FunctionGroupBaseConfig


class MyGroupConfig(FunctionGroupBaseConfig, name="my_group"):
    expose: list[str] = Field(default_factory=list,
                              description="Names of functions to expose globally")
```

## Register the Function Group

Register using {py:deco}`nat.cli.register_workflow.register_function_group`. The registered coroutine should yield a {py:class}`~nat.builder.function.FunctionGroup` instance.

```python
from nat.cli.register_workflow import register_function_group
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup

@register_function_group(config_type=MyGroupConfig)
async def build_my_group(config: MyGroupConfig, builder: Builder):
    group = FunctionGroup(config=config, instance_name="my")

    async def greet_fn(name: str) -> str:
        """Return a friendly greeting."""
        return f"Hello, {name}!"

    async def shout_fn(message: str) -> str:
        """Return the message in uppercase."""
        return message.upper()

    group.add_function(name="greet", fn=greet_fn)
    group.add_function(name="shout", fn=shout_fn)

    yield group
```

## Referencing Functions within a Function Group

- Functions are referenced as `instance_name.function_name` (for example, `my.greet`).
- Only functions listed in `config.expose` are added to the global registry. If `expose` is empty, no functions are globally added, but you can still access the group and its functions programmatically.

```python
async with WorkflowBuilder() as builder:
    await builder.add_function_group("my", MyGroupConfig(expose=["greet"]))

    # Globally exposed
    greet = builder.get_function("my.greet")
    print(await greet.ainvoke("World"))

    my_group = builder.get_function_group("my")

    # You can choose to get the accessible functions directly from the group.
    # If the group has no exposed functions, then this will return all functions in the group.
    # If the group has exposed functions, then this will return only the exposed functions.
    accessible_functions = my_group.get_accessible_functions()

    # Or only the exposed functions (which have also been registered globally as ordinary functions)
    exposed_functions = my_group.get_exposed_functions()
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
- Use `expose` to present only safe, public operations. Keep helper functions unexposed.
- Share expensive resources (for example, clients, caches) through the group context instead of recreating them per function.
- Validate inputs with Pydantic schemas for robust error handling.

## Next Steps

- Review [Writing Custom Functions](./functions.md) for details that also apply to functions inside groups (type safety, streaming vs. single outputs, converters).
