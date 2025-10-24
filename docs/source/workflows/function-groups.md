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

Function groups let you package multiple related functions together so they can share configuration, context, and resources within the NVIDIA NeMo Agent toolkit.

## Overview of Function Groups

By allowing related functions to share a single configuration object and runtime context, function groups solve the following issues you may face when building workflows with multiple functions:

- **Duplicated configuration**: Each function requires the same connection details, credentials, or settings
- **Resource waste**: Creating separate database connections, API clients, or cache instances for each function
- **Scattered logic**: Related operations are defined separately, making code harder to maintain
- **Inconsistent state**: Functions that should share context maintain separate state

### Example: Without Function Groups

Consider three functions that work with an object store. Without function groups, each function needs its own configuration and creates its own connection:

```python
class SaveFileConfig(FunctionBaseConfig, name="save_file"):
    endpoint: str = Field(description="The S3 endpoint URL")
    access_key: str = Field(description="The S3 access key")
    secret_key: str = Field(description="The S3 secret key")
    bucket: str = Field(description="The S3 bucket name")


class LoadFileConfig(FunctionBaseConfig, name="load_file"):
    endpoint: str = Field(description="The S3 endpoint URL")
    access_key: str = Field(description="The S3 access key")
    secret_key: str = Field(description="The S3 secret key")
    bucket: str = Field(description="The S3 bucket name")


class DeleteFileConfig(FunctionBaseConfig, name="delete_file"):
    endpoint: str = Field(description="The S3 endpoint URL")
    access_key: str = Field(description="The S3 access key")
    secret_key: str = Field(description="The S3 secret key")
    bucket: str = Field(description="The S3 bucket name")


@register_function(config_type=SaveFileConfig)
async def build_save_file(config: SaveFileConfig, _builder: Builder):
    # Each function creates its own S3 client
    s3_client = boto3.client('s3', 
                             endpoint_url=config.endpoint,
                             aws_access_key_id=config.access_key,
                             aws_secret_access_key=config.secret_key)
    
    async def save_fn(filename: str, content: bytes) -> str:
        s3_client.put_object(Bucket=config.bucket, Key=filename, Body=content)
        return f"Saved {filename}"
    
    yield save_fn

@register_function(config_type=LoadFileConfig)
async def build_load_file(config: LoadFileConfig, _builder: Builder):
    # Duplicate connection setup
    s3_client = boto3.client('s3',
                             endpoint_url=config.endpoint,
                             aws_access_key_id=config.access_key,
                             aws_secret_access_key=config.secret_key)
    
    async def load_fn(filename: str) -> bytes:
        response = s3_client.get_object(Bucket=config.bucket, Key=filename)
        return response['Body'].read()
    
    yield load_fn

@register_function(config_type=DeleteFileConfig)
async def build_delete_file(config: DeleteFileConfig, _builder: Builder):
    # Yet another duplicate connection
    s3_client = boto3.client('s3',
                             endpoint_url=config.endpoint,
                             aws_access_key_id=config.access_key,
                             aws_secret_access_key=config.secret_key)
    
    async def delete_fn(filename: str) -> str:
        s3_client.delete_object(Bucket=config.bucket, Key=filename)
        return f"Deleted {filename}"
    
    yield delete_fn
```

**Configuration file** (duplicated settings):

```yaml
functions:
  save_file:
    _type: save_file
    endpoint: "https://s3.amazonaws.com"
    access_key: "${S3_ACCESS_KEY}"
    secret_key: "${S3_SECRET_KEY}"
    bucket: "my-bucket"
  
  load_file:
    _type: load_file
    endpoint: "https://s3.amazonaws.com"  # Duplicated
    access_key: "${S3_ACCESS_KEY}"        # Duplicated
    secret_key: "${S3_SECRET_KEY}"        # Duplicated
    bucket: "my-bucket"                   # Duplicated
  
  delete_file:
    _type: delete_file
    endpoint: "https://s3.amazonaws.com"  # Duplicated
    access_key: "${S3_ACCESS_KEY}"        # Duplicated
    secret_key: "${S3_SECRET_KEY}"        # Duplicated
    bucket: "my-bucket"                   # Duplicated
```

**Problems**:
- Three separate S3 clients created
- Configuration repeated three times
- Connection pooling cannot be shared
- Changes require updating three places

### Example: With Function Groups

Using a function group, all three functions share a single S3 client and configuration:

```python
class ObjectStoreConfig(FunctionGroupBaseConfig, name="object_store"):
    endpoint: str = Field(description="The S3 endpoint URL")
    access_key: str = Field(description="The S3 access key")
    secret_key: str = Field(description="The S3 secret key")
    bucket: str = Field(description="The S3 bucket name")


@register_function_group(config_type=ObjectStoreConfig)
async def build_object_store(config: ObjectStoreConfig, _builder: Builder):
    # Create ONE shared S3 client
    s3_client = boto3.client('s3',
                             endpoint_url=config.endpoint,
                             aws_access_key_id=config.access_key,
                             aws_secret_access_key=config.secret_key)
    
    group = FunctionGroup(config=config, instance_name="storage")
    
    async def save_fn(filename: str, content: bytes) -> str:
        s3_client.put_object(Bucket=config.bucket, Key=filename, Body=content)
        return f"Saved {filename}"
    
    async def load_fn(filename: str) -> bytes:
        response = s3_client.get_object(Bucket=config.bucket, Key=filename)
        return response['Body'].read()
    
    async def delete_fn(filename: str) -> str:
        s3_client.delete_object(Bucket=config.bucket, Key=filename)
        return f"Deleted {filename}"
    
    group.add_function(name="save", fn=save_fn, description="Save file to storage")
    group.add_function(name="load", fn=load_fn, description="Load file from storage")
    group.add_function(name="delete", fn=delete_fn, description="Delete file from storage")
    
    yield group
```

**Configuration file** (single configuration):

```yaml
function_groups:
  storage:
    _type: object_store
    endpoint: "https://s3.amazonaws.com"
    access_key: "${S3_ACCESS_KEY}"
    secret_key: "${S3_SECRET_KEY}"
    bucket: "my-bucket"

workflow:
  _type: react_agent
  tool_names: [storage]
  llm_name: my_llm
```

**Benefits**:
- One S3 client shared across all functions
- Configuration defined once
- Connection pooling is efficient
- Changes update in one place
- Functions are all referenced by the group name

## When to Use Function Groups

- **Multiple functions need the same connection** (database, API client, cache)
- **Functions share configuration** (credentials, endpoints, settings)
- **You want to namespace related functions** (`math.add`, `math.multiply`)
- **Functions need to share state** (session data, counters, caches)
- **You have a family of operations** (CRUD operations, data transformations)

## Key Concepts

### Shared Configuration and Context

Function groups are built with a single configuration object and share the runtime context. This enables efficient reuse of connections, caches, and other resources across all functions in the group.

For example, if you create a database connection in your function group, all functions in that group can use the same connection instead of each creating their own.

If we have a collection of math functions, we can create a function group to share the configuration and context for all the functions in the group.

**Python configuration code**:

Without function groups, the configuration types would be:

```python
class AddConfig(FunctionGroupBaseConfig, name="add"):
    rhs: float = Field(description="the number to use as the right-hand-side of the operation")

class MultiplyConfig(FunctionGroupBaseConfig, name="multiply"):
    rhs: float = Field(description="the number to use as the right-hand-side of the operation")

class DivideConfig(FunctionGroupBaseConfig, name="divide"):
    rhs: float = Field(description="the number to use as the right-hand-side of the operation")
```

With function groups, the configuration type is streamlined to:
```python
class MathGroupConfig(FunctionGroupBaseConfig, name="math_group"):
    rhs: float = Field(description="the number to use as the right-hand-side of the operation")
```

**Python implementation code**

Without function groups, we have a lot of duplication:

```python
@register_function_group(config_type=AddConfig)
async def build_add(config: AddConfig, _builder: Builder):
    async def add(a: float) -> float:
        return a + config.rhs

    yield FunctionInfo.from_fn(add, description=f"Adds a number to {config.rhs}")


@register_function_group(config_type=MultiplyConfig)
async def build_add(config: MultiplyConfig, _builder: Builder):
    async def multiply(a: float) -> float:
        return a * config.rhs

    yield FunctionInfo.from_fn(multiply, description=f"Multiplies a number by {config.rhs}")


@register_function_group(config_type=DivideConfig)
async def build_add(config: DivideConfig, _builder: Builder):
    async def divide(a: float) -> float:
        return a / config.rhs

    yield FunctionInfo.from_fn(divide, description=f"Divides a number by {config.rhs}")
```

With function groups, the implementation becomes:

```python
@register_function_group(config_type=MathGroupConfig)
async def build_math_group(config: MathGroupConfig, _builder: Builder):
    # create the function group
    group = FunctionGroup(config=config)

    # define the following operations:
    # - add
    # - multiply
    # - divide
    async def add(a: float) -> float:
        return a + config.rhs
    
    async def multiply(a: float) -> float:
        return a * config.rhs

    async def divide(a: float) -> float:
        if config.rhs == 0:
            raise ValueError("Cannot divide by zero")
        return a / config.rhs

    # add each function to the function group
    group.add_function(name="add", fn=add, description=f"Adds a number to {config.rhs}")
    group.add_function(name="multiply", fn=multiply, description=f"Multiplies a number by {config.rhs}")
    group.add_function(name="divide", fn=divide, description=f"Divides a number by {config.rhs}")

    # return the function group
    # important: must yield rather than return
    yield group
```

**Configuration file**:

Without function groups, the YAML configuration would look like:
```yaml
functions:
  add:
    _type: add
    rhs: 5.0
  multiply:
    _type: multiply
    rhs: 5.0
  divide:
    _type: divide
    rhs: 5.0
```

With function groups, the YAML configuration is simplified to:

```yaml
function_groups:
  math:
    _type: math_group
    rhs: 5.0
```

### Accessing a Function Group

#### From the Configuration File
Accessing a function group from the configuration file is done by its name. This is the same name you use in the `function_groups` section of your workflow configuration.

For example, if your function group is configured as follows:

```yaml
function_groups:
  math:
    _type: math_group
```

You can access it from the configuration file using the name `math`.

#### Programmatically

You can access a function group programmatically using the {py:meth}`~nat.builder.workflow_builder.WorkflowBuilder.get_function_group` method.

```python
math_group = await builder.get_function_group("math")
```

This will return a {py:class}`~nat.builder.function.FunctionGroup` object.

The {py:meth}`~nat.builder.workflow_builder.WorkflowBuilder.get_tools` method can accept a function group name as a tool name in the `tool_names` list.

```python
tools = await builder.get_tools(["math"], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

This will return a list of all accessible functions in the function group that are wrapped for the specified framework.

### Function Naming and Namespacing

Functions inside a group are automatically namespaced by the group instance name. This creates a clear hierarchy and prevents naming conflicts.

**Pattern**: `instance_name.function_name`

**Example**: If your group instance name is `math` and you add functions named `add` and `multiply`:
- Functions become: `math.add` and `math.multiply`
- These names are used in workflow configurations and when calling functions

### Understanding Function Accessibility

Function groups provide different levels of access control. Understanding these levels helps you decide how to configure your function group:

#### Three Levels of Access

- Programmatically Accessible (Always Available)

    All functions added to a function group are always accessible through the group object itself, regardless of include/exclude settings.

    ```python
    # Get the function group
    my_group = await builder.get_function_group("math")

    # Get all functions, even excluded ones
    all_functions = await my_group.get_all_functions()
    ```

- Global Registry (Individually Addressable)

    Functions in the `include` list are added to the global function registry. This means you can:
    - Reference them by their fully qualified name (`math.add`)
    - Use them individually in tool lists
    - Get them directly without accessing the group

    ```python
    # Only works if "add" is in the include list
    add_function = await builder.get_function("math.add")
    ```

- Workflow Builder Tools (Agent-Accessible)

    Functions that are not in the `exclude` list can be wrapped as tools for agents. This makes them:
    - Available to AI agents
    - Discoverable in tool lists
    - Callable by agent frameworks

    ```yaml
    workflow:
      _type: react_agent
      tool_names: [math.add]  # Agent can only use this function (not multiply)
    ```

#### Filtering Functions with `include` and `exclude`

Use these optional configuration fields to control which functions are exposed:

**`include` list**: Explicitly specify which functions should be:
- Added to the global registry (individually addressable)
- Available as workflow tools

```yaml
function_groups:
  math:
    _type: math_group
    include: [add, multiply]  # Only these are globally addressable
```

**`exclude` list**: Specify which functions should NOT be:
- Wrapped as tools for agents
- But they remain programmatically accessible via the function group object

If a function is excluded, it is not added to the global registry and is not available as an accessible tool for agents.

```yaml
function_groups:
  math:
    _type: math_group
    exclude: [divide]  # Make unsafe operations unavailable to agents
```

**Neither specified**: Functions are programmatically accessible through the group but not individually addressable.

:::{note}
`include` and `exclude` are mutually exclusive. Use one or the other, not both.
:::

#### Quick Reference

| Configuration         | Programmatically Accessible | Global Registry     | Agent Tools                 |
|-----------------------|-----------------------------|---------------------|-----------------------------|
| No include/exclude    | ✓ (via group)               | ✗                   | ✓ (all available functions) |
| `include: [add]`      | ✓ (all functions)           | ✓ (only `add`)      | ✓ (only `add`)              |
| `exclude: [divide]`   | ✓ (all functions)           | ✗                   | ✓ (except `divide`)         |

## Using Function Groups

### Creating Custom Function Groups

This section describes how to create and add function groups.

To create your own custom function groups, see the [Writing Custom Function Groups](../extend/function-groups.md) guide, which covers:

- Defining configuration classes with Pydantic fields
- Registering function groups with decorators
- Implementing builder functions
- Sharing resources with context managers (for example, database connections and API clients)
- Customizing input schemas for better validation
- Implementing dynamic filtering for runtime control
- Best practices, common patterns, and troubleshooting

The rest of this guide focuses on **using existing function groups** in your workflows.

### Adding a Function Group to a Workflow

The `function_groups` section of a workflow configuration declares groups by instance name and type. The `workflow.tool_names` field can reference either the entire group or individual functions.

#### Example 1: Using the Entire Group (Simplest)

The simplest configuration references the entire function group, making all its functions available to the agent:

```yaml
function_groups:
  math:
    _type: math_group

workflow:
  _type: react_agent
  tool_names: [math]
  llm_name: my_llm
```

All functions in the `math` group (`math.add`, `math.multiply`) become available as tools for the agent.

#### Example 2: Including Specific Functions

Use the `include` list to control which functions are individually addressable and wrapped as tools:

```yaml
function_groups:
  math:
    _type: math_group
    include: [add, multiply]

workflow:
  _type: react_agent
  tool_names: [math.add, math.multiply]
  llm_name: my_llm
```

Now you can reference individual functions in `tool_names`. Only included functions are added to the global registry.

#### Example 3: Excluding Specific Functions

Use the `exclude` list to prevent certain functions from being exposed to agents:

```yaml
function_groups:
  math:
    _type: math_group
    exclude: [divide]  # Exclude division to prevent divide-by-zero issues

workflow:
  _type: react_agent
  tool_names: [math]
  llm_name: my_llm
```

All functions except `divide` are available to the agent. Functions are not in the global registry and are not individually addressable. The excluded function remains programmatically accessible using the function group object.

#### Example 4: Mixing Group and Individual References

You can reference some function groups as a whole and others individually:

```yaml
function_groups:
  math:
    _type: math_group
    include: [add, multiply, divide]
  
  storage:
    _type: object_store
    endpoint: "https://s3.amazonaws.com"
    bucket: "my-bucket"

workflow:
  _type: react_agent
  tool_names: [math.add, storage]  # Individual function + whole group
  llm_name: my_llm
```


### Using Function Groups Programmatically

You can work with function groups directly in Python code using the {py:class}`~nat.builder.workflow_builder.WorkflowBuilder`.

#### Adding a Function Group

```python
from nat.builder.workflow_builder import WorkflowBuilder

async with WorkflowBuilder() as builder:
    # Add the function group
    await builder.add_function_group("math", MathGroupConfig(rhs=5.0, include=["add", "multiply"]))
    
    # Call an included function by its fully-qualified name
    add = await builder.get_function("math.add")
    result = await add.ainvoke(3.0)  # Returns: 8.0
```

#### Getting the Function Group Object

Access the function group object to work with all functions, including excluded ones:

```python
async with WorkflowBuilder() as builder:
    await builder.add_function_group("math", MathGroupConfig(exclude=["divide"]))
    
    # Get the function group object
    math_group = await builder.get_function_group("math")
    
    # Get all accessible functions (respects include/exclude)
    accessible = await math_group.get_accessible_functions()
    
    # Get all functions including excluded ones
    all_funcs = await math_group.get_all_functions()
    
    # Get only included functions
    included = await math_group.get_included_functions()
    
    # Get only excluded functions
    excluded = await math_group.get_excluded_functions()
```

#### Getting Tools for Agent Frameworks

To wrap all accessible functions in a group for a specific agent framework:

```python
from nat.data_models.component_ref import FunctionGroupRef
from nat.builder.framework_enum import LLMFrameworkEnum

async with WorkflowBuilder() as builder:
    await builder.add_function_group("math", MathGroupConfig(include=["add", "multiply"]))

    # Get tools wrapped for the specified framework
    tools = await builder.get_tools(["math"], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

## Advanced Features

Use advanced features like dynamic, group-level, and per-function filters to control which functions are accessible at runtime.

### Dynamic Filtering

Function groups support dynamic filtering to control which functions are accessible at runtime. Filters work alongside the `include` and `exclude` configuration and are applied when functions are accessed.

#### Group-Level Filters

Group-level filters receive a list of function names and return a filtered list:

```python
async with WorkflowBuilder() as builder:
    # Define a filter that only allows "add" operations
    def math_filter(function_names):
        return [name for name in function_names if name.startswith("add")]
    
    # Add the function group
    config = MathGroupConfig(include=["add", "multiply"])
    await builder.add_function_group("math", config)
    
    # Apply the filter
    math_group = await builder.get_function_group("math")
    math_group.set_filter_fn(math_filter)
    
    # Now only "add" functions are accessible
    accessible = await math_group.get_accessible_functions()
    # Returns: ["math.add"]
```

#### Per-Function Filters

Per-function filters are applied to individual functions during group creation. See the [Writing Custom Function Groups](../extend/function-groups.md) guide for details.

#### Filter Interaction

Filters work in combination with `include` and `exclude` configuration as described in the following workflow:

1. Configuration filtering is applied first (`include`/`exclude`)
2. Group-level filters are applied to the result
3. Per-function filters are applied to each remaining function

## Best Practices

This section describes best practices when using function groups.

### When to Use Function Groups

**Use function groups when you have**:
- Multiple functions that need the same database connection, API client, or cache
- Related operations that share configuration (credentials, endpoints, timeouts)
- A family of functions that benefit from namespacing (CRUD operations, math operations)
- Functions that need to share state or context

**Use individual functions when**:
- Each function is completely independent
- Functions have no shared resources
- You only need one or two simple functions
- The overhead of creating a group isn't justified

### Common Patterns
This section describes common patterns when using function groups.

#### Pattern 1: Database Operations

Group all database operations together to share a connection pool:

```python
@register_function_group(config_type=DatabaseConfig)
async def build_database_group(config: DatabaseConfig, _builder: Builder):
    async with create_connection_pool(config) as pool:
        group = FunctionGroup(config=config, instance_name="db")
        
        # All functions share the same pool
        group.add_function("query", query_fn)
        group.add_function("insert", insert_fn)
        group.add_function("update", update_fn)
        group.add_function("delete", delete_fn)
        
        yield group
```

#### Pattern 2: API Client Operations

Group API calls that use the same authentication and base URL:

```python
@register_function_group(config_type=APIConfig)
async def build_api_group(config: APIConfig, _builder: Builder):
    # One authenticated client for all operations
    client = httpx.AsyncClient(
        base_url=config.base_url,
        headers={"Authorization": f"Bearer {config.api_key}"}
    )
    
    group = FunctionGroup(config=config, instance_name="api")
    
    # All functions use the same authenticated client
    group.add_function("get_user", get_user_fn)
    group.add_function("list_items", list_items_fn)
    group.add_function("create_item", create_item_fn)
    
    yield group
    await client.aclose()
```

#### Pattern 3: Partial Exposure with Exclude

Expose most functions but keep internal helpers private:

```python
function_groups:
  math:
    _type: math_group
    exclude: [_internal_helper, _validate_input]  # Keep helpers private
    
workflow:
  _type: react_agent
  tool_names: [math]  # Agents get public functions only
```

#### Pattern 4: Selective Exposure with Include

Only expose safe or tested functions:

```python
function_groups:
  experimental:
    _type: ml_models
    include: [stable_model_v1]  # Only expose production-ready models
    
workflow:
  _type: react_agent
  tool_names: [experimental.stable_model_v1]
```

### Configuration Best Practices
Ensure you adhere to the configuration best practices when using function groups.
#### Keep Instance Names Short

Instance names become part of function names, so keep them concise:

```python
# Good
group = FunctionGroup(config=config, instance_name="db")
# Results in: db.query, db.insert

# Less ideal
group = FunctionGroup(config=config, instance_name="database_operations")
# Results in: database_operations.query, database_operations.insert
```

#### Use Environment Variables for Secrets

Never embed credentials in configuration files:

```yaml
function_groups:
  storage:
    _type: object_store
    endpoint: "${S3_ENDPOINT}"
    access_key: "${S3_ACCESS_KEY}"
    secret_key: "${S3_SECRET_KEY}"
```

#### Provide Sensible Defaults

Make configuration optional when reasonable defaults exist:

```python
class CacheGroupConfig(FunctionGroupBaseConfig, name="cache_group"):
    ttl: int = Field(default=3600, description="Cache time-to-live in seconds")
    max_size: int = Field(default=1000, description="Maximum cache entries")
```

### Resource Management

#### Always Use Context Managers for Resources

Ensure proper cleanup of connections and resources:

```python
# Good
async with create_pool(config) as pool:
    group = FunctionGroup(config=config, instance_name="db")
    # Add functions
    yield group
# Pool closes automatically

# Bad - resource may leak
pool = create_pool(config)
group = FunctionGroup(config=config, instance_name="db")
yield group
```

#### Share Expensive Resources

Create resources once and share them across all functions:

```python
# Good - one shared client
@register_function_group(config_type=Config)
async def build_group(config: Config, _builder: Builder):
    client = expensive_client_setup()
    # All functions use the same client
    
# Bad - each function creates its own client
async def fn1():
    client = expensive_client_setup()
    
async def fn2():
    client = expensive_client_setup()  # Wasteful duplication
```

### Anti-Patterns to Avoid

#### Don't Use Function Groups for Unrelated Functions

```python
# Bad - mixing unrelated concerns
group = FunctionGroup(config=config, instance_name="utils")
group.add_function("database_query", db_fn)
group.add_function("send_email", email_fn)
group.add_function("calculate_tax", tax_fn)
```

Instead, create separate groups for different concerns or use individual functions.

#### Don't Create Groups for Single Functions

```python
# Bad - unnecessary overhead
@register_function_group(config_type=Config)
async def build_group(config: Config, _builder: Builder):
    group = FunctionGroup(config=config, instance_name="single")
    group.add_function("only_one", fn)
    yield group
```

Use `@register_function` for single functions instead.

#### Don't Recreate Resources Per Function

```python
# Bad - defeats the purpose of function groups
@register_function_group(config_type=Config)
async def build_group(config: Config, _builder: Builder):
    group = FunctionGroup(config=config, instance_name="db")
    
    async def query_fn():
        conn = create_connection()  # Bad - creates new connection each time
        
    group.add_function("query", query_fn)
    yield group
```

Create the resource once outside the functions.

#### Don't Use Both Include and Exclude

```yaml
# Bad - these are mutually exclusive
function_groups:
  math:
    _type: math_group
    include: [add, multiply]
    exclude: [divide]  # Error!
```

Choose one or the other based on your needs.

### Testing Considerations

When testing workflows with function groups:

```python
# Test individual functions through the group
async with WorkflowBuilder() as builder:
    await builder.add_function_group("math", MathGroupConfig(rhs=5.0))
    math_group = await builder.get_function_group("math")
    
    # Test each function
    all_funcs = await math_group.get_all_functions()
    for func_name, func in all_funcs.items():
        result = await func.ainvoke(test_input)
        assert result == expected_output
```

## Writing Function Groups

For details on creating and registering your own groups, see the [Writing Custom Function Groups](../extend/function-groups.md) guide.
