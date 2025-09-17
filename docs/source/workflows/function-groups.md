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

# Function Groups

Function groups let you package multiple related functions together so they can share configuration, context, and resources within the NeMo Agent toolkit.

In a function group, you define several callable operations (functions) and optionally choose which ones to include or exclude. Explicitly included functions become addressable by fully qualified names like `group_name.function_name` and can be wrapped as tools for agent frameworks. Excluded functions are still accessible programmatically, but are not addressable by fully qualified names or wrapped as tools by default.

## Key Concepts

### Shared Configuration and Context

Function groups are built with a single configuration object and share the runtime context. This enables efficient reuse of connections, caches, and other resources across all functions in the group.

### Name Grouping

Functions inside a group are automatically grouped by the group instance name. If the group instance name is `math`, and you add a function named `add`, the qualified function name becomes `math.add`.

### Exposing Functions

Use the `include` list to control which functions are added to the global registry. If `include` is empty, the group remains usable as a group, but individual functions are not globally addressable. If `exclude` is provided, matching functions are filtered out and are not wrapped as tools, but they remain accessible programmatically.

### Dynamic Filtering

Function groups support dynamic filtering to control which functions are accessible at runtime. Filters work alongside the `include` and `exclude` configuration and are applied when functions are accessed, not when they are added.

#### Filter Types

**Group-level filters**: Applied to all functions in the group and receive a list of function names to filter.

**Per-function filters**: Applied to individual functions and receive the function name to determine inclusion.

#### Filter Interaction

Filters work in combination with `include` and `exclude` configuration:

1. First, the configuration determines the base set of functions (include/exclude logic)
2. Then, group-level filters are applied to the resulting set
3. Finally, per-function filters are applied to each remaining function

### Tool Wrapping

You can request tools for an entire function group. The builder will wrap all accessible functions (honoring `include`, `exclude`, and any active filters) into the requested tool wrapper for a given agent framework.

## Using Function Groups

### Registering a Function Group

Register a function group with the {py:deco}`nat.cli.register_workflow.register_function_group` decorator. The builder expects the registered coroutine to yield a {py:class}`~nat.builder.function.FunctionGroup` instance.

```python
from pydantic import Field
from nat.data_models.function import FunctionGroupBaseConfig
from nat.cli.register_workflow import register_function_group
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup


class MathGroupConfig(FunctionGroupBaseConfig, name="math_group"):
    # If you want to provide a default include list, you can do so here.
    # Otherwise, you can specify the include list in the workflow configuration.
    # If you do not provide an include list, then you cannot reference individual functions in the tool_names list.
    # You can still reference the entire group.

    # include: list[str] = Field(default_factory=lambda: ["add"],
    #                          description="Functions to include globally")

    # OR

    # exclude: list[str] = Field(default_factory=lambda: ["mul"],
    #                          description="Functions to exclude globally")
    pass


@register_function_group(config_type=MathGroupConfig)
async def build_math_group(config: MathGroupConfig, _builder: Builder):

    # context will be shared across all functions in the group
    async with SomeAsyncContextManager() as context:

        group = FunctionGroup(config=config, instance_name="math")

        async def add_fn(values: list[int]) -> int:
            """Add a list of integers and return the sum."""
            return sum(values)

        async def mul_fn(values: list[int]) -> int:
            """Multiply a list of integers and return the product."""
            result = 1
            for v in values:
                result *= v
            return result

        group.add_function(name="add", fn=add_fn, description=add_fn.__doc__)
        group.add_function(name="mul", fn=mul_fn, description=mul_fn.__doc__)

        # note that the group is yielded within the context manager
        yield group
```

### Adding a Function Group to a Workflow

The `function_groups` section of a workflow configuration declares groups by instance name and type, and the `workflow.tool_names` field can reference either the entire group or individual functions.

If you do not provide an `include` list, then you cannot reference individual functions in the `tool_names` list. You can still reference the entire group.

```yaml
general:
  # ...

function_groups:
  math:
    _type: math_group
    # Option A: include and exclude no functions globally, but still usable as a group
    # # omit include and exclude lists

    # Option B: exclude selected functions globally
    exclude: [mul]

    # Option C: include selected functions globally (accessible as a function named math.add, math.mul)
    include: [add, mul]

workflow:
  _type: react_agent
  # Option A: reference the group (only non-excluded functions are accessible)
  # tool_names: [math]  # note that this is the group name, not the function name

  # Option B: reference specific functions (can only be used if include list is provided)
  tool_names: [math.add]

  llm_name: my_llm
  verbose: true
```


#### Adding a Function Group to a Workflow Programmatically

Use {py:meth}`~nat.builder.workflow_builder.WorkflowBuilder.add_function_group` to add a group to your workflow. Included functions are automatically added to the global function registry with names like `math.add`. Excluded functions are not added to the global function registry and are not accessible by fully qualified names or wrapped as tools by default. However, they are still accessible programmatically.

```python
from nat.builder.workflow_builder import WorkflowBuilder


async with WorkflowBuilder() as builder:
    # Add the function group; only included functions become globally available
    await builder.add_function_group("math", MathGroupConfig(include=["add"]))

    # Call an included function directly by its fully-qualified name
    add = builder.get_function("math.add")
    result = await add.ainvoke([1, 2, 3])  # 6
```

#### Using Filters

You can apply dynamic filters to control function accessibility at runtime:

```python
async with WorkflowBuilder() as builder:
    # Add function group with a group-level filter
    def math_filter(function_names):
        # Only allow functions that start with "add"
        return [name for name in function_names if name.startswith("add")]
    
    config = MathGroupConfig(include=["add", "mul"])
    group = await builder.add_function_group("math", config)
    
    # Set the group-level filter
    math_group = builder.get_function_group("math")
    math_group.set_filter_fn(math_filter)
    
    # Now only "add" functions will be accessible, even though "mul" was included
    accessible = math_group.get_accessible_functions()
    # accessible contains only "math.add"
```

### Getting Tools for a Function Group

To wrap all accessible functions in a group for a specific agent framework, request tools with the group name. The builder resolves the group and returns wrapped tools for each accessible function.

```python
from nat.data_models.component_ref import FunctionGroupRef
from nat.builder.framework_enum import LLMFrameworkEnum


tools = builder.get_tools([FunctionGroupRef("math")], wrapper_type=LLMFrameworkEnum.LANGCHAIN)

# or the following:
# tools = builder.get_tools(["math"], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

## Writing Function Groups

For details on creating and registering your own groups, see the [Writing Custom Function Groups](../extend/function-groups.md) guide.
