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

:::{note}
Before creating your own function groups, ensure you read the [Function Groups](../workflows/function-groups.md) guide to understand how function groups work.
:::

This guide shows you how to create custom function groups for the NVIDIA NeMo Agent toolkit. Function groups bundle related functions that share configuration, resources, and runtime context.

## When to Write a Custom Function Group

Create a custom function group when you need to:

- **Share expensive resources**: Database connections, API clients, cache instances, or connection pools across multiple related functions
- **Bundle related operations**: Group CRUD operations, file operations, or API endpoints that belong together
- **Centralize configuration**: Manage credentials, endpoints, and settings in one place for multiple functions
- **Create reusable components**: Package functionality that can be used across multiple workflows
- **Namespace functions**: Organize functions into logical groups, such as `db.query`, `db.insert`, `api.get`, and `api.post`

## Step 1: Define the Configuration

Every function group needs a configuration class that inherits from {py:class}`~nat.data_models.function.FunctionGroupBaseConfig`.

### Minimal Configuration

Start with the simplest possible configuration:

```python
from nat.data_models.function import FunctionGroupBaseConfig

class MyGroupConfig(FunctionGroupBaseConfig, name="my_group"):
    """Configuration for my custom function group."""
    pass
```

The `name` parameter (`my_group`) is the type identifier used in YAML configurations as `_type: my_group`.

### Adding Configuration Fields

Add fields for any settings your functions need to share:

```python
from pydantic import Field
from nat.data_models.function import FunctionGroupBaseConfig

class DatabaseGroupConfig(FunctionGroupBaseConfig, name="database_group"):
    """Configuration for database operations."""
    host: str = Field(description="Database host address")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(description="Database name")
    user: str = Field(description="Database user")
    password: str = Field(description="Database password")
    max_connections: int = Field(default=10, description="Maximum pool size")
```

These fields become available in your YAML configuration:

```yaml
function_groups:
  db:
    _type: database_group
    host: "localhost"
    port: 5432
    database: "mydb"
    user: "${DB_USER}"
    password: "${DB_PASSWORD}"
    max_connections: 20
```

### Controlling Function Exposure

The {py:class}`~nat.data_models.function.FunctionGroupBaseConfig` configuration class has two optional fields: `include` and `exclude`. These fields are used to control which functions are exposed through the function group or excluded from the function group.

If your function group is intended to override the default behavior of the function group, you can use the `include` field to specify which functions to expose and the `exclude` field to specify which functions to exclude.

If your function group is intended to be a simple wrapper around a set of functions, you can omit both fields and all functions will be exposed through the function group.

```python
class APIGroupConfig(FunctionGroupBaseConfig, name="api_group"):
    """Configuration for API operations."""
    base_url: str = Field(description="API base URL")
    api_key: str = Field(description="API authentication key")
    
    # Optional: specify which functions to expose
    include: list[str] = Field(
        default_factory=list,
        description="Functions to expose globally"
    )
    
    # Or alternatively, specify which to hide
    exclude: list[str] = Field(
        default_factory=list,
        description="Functions to keep private"
    )
```

:::{note}
`include` and `exclude` are mutually exclusive. If both are provided, a `ValueError` will be raised.
:::

When to use `include`, `exclude`, or neither:
- Use `include` when you want to explicitly list exposed functions (allowlist approach)
- Use `exclude` when most functions are public but some are private (blocklist approach)
- Omit both when all functions should be accessible through the group reference only

## Step 2: Register and Implement the Function Group

Use the {py:deco}`~nat.cli.register_workflow.register_function_group` decorator to register your function group builder.

### Basic Implementation

Here's the simplest function group implementation:

```python
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

class MyGroupConfig(FunctionGroupBaseConfig, name="my_group"):
    """Configuration for my custom function group."""
    pass

@register_function_group(config_type=MyGroupConfig)
async def build_my_group(config: MyGroupConfig, _builder: Builder):
    # Create the function group with an instance name
    group = FunctionGroup(config=config, instance_name="my")
    
    # Define your functions
    async def greet_fn(name: str) -> str:
        """Return a friendly greeting given a name."""
        return f"Hello, {name}!"
    
    async def farewell_fn(name: str) -> str:
        """Return a farewell message given a name."""
        return f"Goodbye, {name}!"
    
    # Add functions to the group
    group.add_function(name="greet", fn=greet_fn, description=greet_fn.__doc__)
    group.add_function(name="farewell", fn=farewell_fn, description=farewell_fn.__doc__)
    
    # Yield the group to make it available
    yield group
```

**Key components**:
- **Decorator**: `@register_function_group(config_type=MyGroupConfig)` registers the builder
- **Instance name**: `instance_name="my"` creates the namespace (`my.greet`, `my.farewell`)
- **Function definitions**: Define async functions that implement your logic
- **Add to group**: Use `group.add_function()` to register each function
- **Yield**: `yield group` makes the group available to workflows

### Using Configuration Values

Access configuration values in your functions to customize behavior:

```python
import httpx
from nat.cli.register_workflow import register_function_group

@register_function_group(config_type=APIGroupConfig)
async def build_api_group(config: APIGroupConfig, _builder: Builder):
    # Create authenticated HTTP client using config
    async with httpx.AsyncClient(
        base_url=config.base_url,
        headers={"Authorization": f"Bearer {config.api_key}"},
        timeout=30.0
    ) as client:
        group = FunctionGroup(config=config, instance_name="api")
        
        async def get_user_fn(user_id: int) -> dict:
            """Get user details by ID."""
            response = await client.get(f"/users/{user_id}")
            response.raise_for_status()
            return response.json()
        
        async def create_item_fn(name: str, description: str) -> dict:
            """Create a new item."""
            response = await client.post(
                "/items",
                json={"name": name, "description": description}
            )
            response.raise_for_status()
            return response.json()
        
        group.add_function(name="get_user", fn=get_user_fn, description=get_user_fn.__doc__)
        group.add_function(name="create_item", fn=create_item_fn, description=create_item_fn.__doc__)
        
        yield group
```

### Sharing Resources with Context Managers

For functions that need shared resources (for example, connections and clients), use context managers:

```python
import asyncpg

from nat.cli.register_workflow import register_function_group
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup

@register_function_group(config_type=DatabaseGroupConfig)
async def build_database_group(config: DatabaseGroupConfig, _builder: Builder):
    # Create a shared connection pool
    async with asyncpg.create_pool(
        host=config.host,
        port=config.port,
        database=config.database,
        user=config.user,
        password=config.password,
        min_size=1,
        max_size=config.max_connections
    ) as pool:
        # Create the function group
        group = FunctionGroup(config=config, instance_name="db")
        
        # All functions can access the shared pool
        async def query_fn(sql: str) -> list[dict]:
            """Execute a SQL query and return results as dictionaries."""
            async with pool.acquire() as conn:
                rows = await conn.fetch(sql)
                return [dict(row) for row in rows]
        
        async def execute_fn(sql: str) -> str:
            """Execute a SQL statement (INSERT, UPDATE, DELETE)."""
            async with pool.acquire() as conn:
                await conn.execute(sql)
                return "Statement executed successfully"
        
        async def count_fn(table: str) -> int:
            """Count rows in a table."""
            async with pool.acquire() as conn:
                result = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                return result
        
        # Add all functions to the group
        group.add_function(name="query", fn=query_fn, description=query_fn.__doc__)
        group.add_function(name="execute", fn=execute_fn, description=execute_fn.__doc__)
        group.add_function(name="count", fn=count_fn, description=count_fn.__doc__)
        
        # Yield within the context manager to keep pool alive
        yield group
    # Pool automatically closes when workflow ends
```

**Why use context managers**:
- Resource lifecycle management (automatic cleanup)
- Connection pooling efficiency
- Proper error handling
- Prevents resource leaks

## Step 3: Customize Function Schemas

The toolkit automatically infers input and output schemas from your function type hints. You can customize these schemas for better validation and documentation. See the [Writing Custom Functions](./functions.md) guide for more information.

## Step 4: Work with Function Groups Programmatically

After creating your function group, you can work with it programmatically in your workflows.

### Accessing Functions

Functions are referenced as `instance_name.function_name`:

```python
from nat.builder.workflow_builder import WorkflowBuilder

async with WorkflowBuilder() as builder:
    # Add the function group with specific functions included
    await builder.add_function_group("my", MyGroupConfig(include=["greet", "farewell"]))
    
    # Access individual function by fully qualified name
    greet = await builder.get_function("my.greet")
    result = await greet.ainvoke("World")
    print(result)  # "Hello, World!"
```

### Getting Functions from the Group

Access the function group object to work with multiple functions:

```python
async with WorkflowBuilder() as builder:
    await builder.add_function_group("my", MyGroupConfig(include=["greet"]))
    
    # Get the function group object
    my_group = await builder.get_function_group("my")
    
    # Get accessible functions (respects include/exclude)
    accessible = await my_group.get_accessible_functions()
    # Returns: {"greet": <function>}
    
    # Get all functions (ignores include/exclude)
    all_funcs = await my_group.get_all_functions()
    # Returns: {"greet": <function>, "farewell": <function>}
    
    # Get only included functions
    included = await my_group.get_included_functions()
    # Returns: {"greet": <function>}
    
    # Get only excluded functions
    excluded = await my_group.get_excluded_functions()
    # Returns: {"farewell": <function>}
```

### Testing Your Function Group

Test individual functions through the group:

```python
import pytest
from nat.builder.workflow_builder import WorkflowBuilder

@pytest.mark.asyncio
async def test_my_function_group():
    async with WorkflowBuilder() as builder:
        await builder.add_function_group("my", MyGroupConfig())
        my_group = await builder.get_function_group("my")
        
        # Test each function
        all_funcs = await my_group.get_all_functions()
        
        # Test greet function
        greet = all_funcs["greet"]
        result = await greet.ainvoke("Alice")
        assert result == "Hello, Alice!"
        
        # Test farewell function
        farewell = all_funcs["farewell"]
        result = await farewell.ainvoke("Bob")
        assert result == "Goodbye, Bob!"
```

## Step 5: Advanced - Dynamic Filtering (Optional)

Dynamic filters provide runtime control over which functions are accessible. Use filters when function availability needs to depend on runtime conditions like environment, feature flags, or user permissions.

:::{note}
Most function groups don't need filters. Use `include`/`exclude` lists for static function control. Only use filters when you need dynamic runtime behavior.
:::

### When to Use Filters

**Use filters for**:
- Environment-based function availability (development vs. production)
- Feature flags that change at runtime
- User permission-based access control
- A/B testing different function sets

**Use include/exclude for**:
- Static function exposure that doesn't change
- Hiding internal helper functions
- Permanently excluding unsafe operations

### Group-Level Filters

Group-level filters receive a list of function names and return a filtered list:

```python
from collections.abc import Sequence
from nat.cli.register_workflow import register_function_group
from nat.builder.function import FunctionGroup

class EnvironmentGroupConfig(FunctionGroupBaseConfig, name="env_group"):
    """Configuration with environment setting."""
    environment: str = Field(default="development", description="Deployment environment")

@register_function_group(config_type=EnvironmentGroupConfig)
async def build_env_group(config: EnvironmentGroupConfig, _builder: Builder):
    # Define a group-level filter based on environment
    async def environment_filter(function_names: Sequence[str]) -> Sequence[str]:
        """Only expose admin functions in development."""
        if config.environment == "production":
            # In production, exclude admin functions
            return [name for name in function_names if not name.startswith("admin_")]
        # In development, allow all functions
        return function_names
    
    # Create group with the filter
    group = FunctionGroup(config=config, instance_name="ops", filter_fn=environment_filter)
    
    # Add admin and user functions
    async def admin_reset_fn() -> str:
        """Reset system (admin only)."""
        return "System reset"
    
    async def admin_config_fn(key: str, value: str) -> str:
        """Update config (admin only)."""
        return f"Config updated: {key}={value}"
    
    async def user_status_fn() -> dict:
        """Get system status (available to all)."""
        return {"status": "healthy", "uptime": 12345}
    
    group.add_function("admin_reset", admin_reset_fn, description=admin_reset_fn.__doc__)
    group.add_function("admin_config", admin_config_fn, description=admin_config_fn.__doc__)
    group.add_function("user_status", user_status_fn, description=user_status_fn.__doc__)
    
    yield group
```

**Result**:
- Development: All three functions available
- Production: Only `user_status` available (admin functions filtered out)

### Per-Function Filters

Per-function filters are applied to individual functions and determine whether that specific function should be included:

```python
class FeatureFlagConfig(FunctionGroupBaseConfig, name="feature_flag_group"):
    enable_experimental: bool = Field(default=False, description="Enable experimental features")
    enable_beta: bool = Field(default=False, description="Enable beta features")

@register_function_group(config_type=FeatureFlagConfig)
async def build_feature_group(config: FeatureFlagConfig, _builder: Builder):
    group = FunctionGroup(config=config, instance_name="features")
    
    # Filters for different feature types
    async def experimental_only(name: str) -> bool:
        """Only include if experimental features are enabled."""
        return config.enable_experimental
    
    async def beta_only(name: str) -> bool:
        """Only include if beta features are enabled."""
        return config.enable_beta
    
    # Stable function (always available)
    async def stable_feature_fn() -> str:
        """A stable, production-ready feature."""
        return "Stable feature"
    
    # Beta function (conditionally available)
    async def beta_feature_fn() -> str:
        """A beta feature under testing."""
        return "Beta feature"
    
    # Experimental function (conditionally available)
    async def experimental_feature_fn() -> str:
        """An experimental feature in early development."""
        return "Experimental feature"
    
    # Add functions with appropriate filters
    group.add_function("stable", stable_feature_fn, description=stable_feature_fn.__doc__)
    group.add_function("beta", beta_feature_fn, description=beta_feature_fn.__doc__, 
                      filter_fn=beta_only)
    group.add_function("experimental", experimental_feature_fn, 
                      description=experimental_feature_fn.__doc__, 
                      filter_fn=experimental_only)
    
    yield group
```

**Configuration in YAML**:
```yaml
function_groups:
  features:
    _type: feature_flag_group
    enable_experimental: false  # Experimental functions hidden
    enable_beta: true           # Beta functions available
```

### Filter Execution Order

Filters work in combination with `include` and `exclude` configuration in a specific order:

1. **Configuration filtering** (`include`/`exclude` lists) - applied first
2. **Group-level filtering** - applied to the result of step 1
3. **Per-function filtering** - applied to each function from step 2

**Example**:

```python
from collections.abc import Sequence

class ComplexFilterConfig(FunctionGroupBaseConfig, name="complex_filter_group"):
    include: list[str] = Field(default_factory=lambda: ["func1", "func2", "func3", "test_func4"])
    environment: str = Field(default="development")
    enable_experimental: bool = Field(default=False)

@register_function_group(config_type=ComplexFilterConfig)
async def build_complex_group(config: ComplexFilterConfig, _builder: Builder):
    # Group-level filter: Remove test functions in production
    async def env_filter(names: Sequence[str]) -> Sequence[str]:
        if config.environment == "production":
            return [name for name in names if not name.startswith("test_")]
        return names
    
    # Per-function filter: Only include experimental if flag is set
    async def experimental_gate(name: str) -> bool:
        return config.enable_experimental
    
    group = FunctionGroup(config=config, filter_fn=env_filter)
    
    # Add functions
    group.add_function("func1", fn1)           # Always included
    group.add_function("func2", fn2)           # Always included
    group.add_function("func3_experimental",   # Conditionally included
                      fn3, 
                      filter_fn=experimental_gate)
    group.add_function("test_func4", fn4)     # Removed in production by group filter
    group.add_function("func5", fn5)          # NOT in include list, so never accessible
    
    yield group
```

**Result in production** (`environment="production"`, `enable_experimental=False`):
1. Start with: `["func1", "func2", "func3_experimental", "test_func4"]` (include list)
2. After group filter: `["func1", "func2", "func3_experimental"]` (test_func4 removed)
3. After per-function filter: `["func1", "func2"]` (func3_experimental removed)

**Result in development** (`environment="development"`, `enable_experimental=True`):
1. Start with: `["func1", "func2", "func3_experimental", "test_func4"]` (include list)
2. After group filter: `["func1", "func2", "func3_experimental", "test_func4"]` (all pass)
3. After per-function filter: `["func1", "func2", "func3_experimental", "test_func4"]` (all pass)

## Common Patterns

### Pattern 1: Database Connection Pool

```python
@register_function_group(config_type=DatabaseConfig)
async def build_db_group(config: DatabaseConfig, _builder: Builder):
    async with asyncpg.create_pool(...) as pool:
        group = FunctionGroup(config=config, instance_name="db")
        
        async def query(sql: str) -> list[dict]:
            async with pool.acquire() as conn:
                return [dict(r) for r in await conn.fetch(sql)]
        
        async def execute(sql: str) -> int:
            async with pool.acquire() as conn:
                result = await conn.execute(sql)
                return int(result.split()[-1])  # Return affected rows
        
        group.add_function("query", query)
        group.add_function("execute", execute)
        yield group
```

### Pattern 2: Authenticated API Client

```python
@register_function_group(config_type=APIConfig)
async def build_api_group(config: APIConfig, _builder: Builder):
    headers = {"Authorization": f"Bearer {config.api_key}"}
    async with httpx.AsyncClient(base_url=config.base_url, headers=headers) as client:
        group = FunctionGroup(config=config, instance_name="api")
        
        async def get(endpoint: str) -> dict:
            response = await client.get(endpoint)
            response.raise_for_status()
            return response.json()
        
        async def post(endpoint: str, data: dict) -> dict:
            response = await client.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        
        group.add_function("get", get)
        group.add_function("post", post)
        yield group
```

### Pattern 3: Stateful Cache

```python
@register_function_group(config_type=CacheConfig)
async def build_cache_group(config: CacheConfig, _builder: Builder):
    # Shared cache state
    cache: dict[str, tuple[Any, float]] = {}
    
    group = FunctionGroup(config=config, instance_name="cache")
    
    async def set_value(key: str, value: Any) -> str:
        """Set a cache value with TTL."""
        cache[key] = (value, time.time() + config.ttl)
        return f"Cached: {key}"
    
    async def get_value(key: str) -> Any | None:
        """Get a cache value if not expired."""
        if key in cache:
            value, expires = cache[key]
            if time.time() < expires:
                return value
            del cache[key]
        return None
    
    async def clear_cache() -> str:
        """Clear all cache entries."""
        cache.clear()
        return "Cache cleared"
    
    group.add_function("set", set_value)
    group.add_function("get", get_value)
    group.add_function("clear", clear_cache)
    yield group
```

## Troubleshooting

### Issue: Functions Not Appearing in Workflow

**Problem**: Functions are not available even though they are added to the group.

**Solution**: Check your `include` list configuration:
```yaml
function_groups:
  mygroup:
    _type: my_group
    include: [func1, func2]  # Must list functions explicitly
```

Or reference the entire group:
```yaml
workflow:
  tool_names: [mygroup]  # Use group name, not individual functions
```

### Issue: Resource Leaks

**Problem**: Database connections or other resources are not being cleaned up.

**Solution**: Always yield within the context manager:
```python
# Correct
async with create_pool() as pool:
    group = FunctionGroup(...)
    yield group  # Inside context

# Wrong
async with create_pool() as pool:
    group = FunctionGroup(...)
yield group  # Outside context - pool already closed!
```

### Issue: Filter Not Working

**Problem**: Filter function is not affecting available functions.

**Solution**: Ensure filter is set before accessing functions and check the filter logic:
```python
# Make sure filter is applied to the group
group = FunctionGroup(config=config, instance_name="my", filter_fn=my_filter)

# Or set it after creation
group.set_filter_fn(my_filter)

# Debug: Check what's being filtered
accessible = await group.get_accessible_functions()
print(f"Accessible functions: {list(accessible.keys())}")
```

### Issue: Type Validation Errors

**Problem**: Function input validation fails unexpectedly.

**Solution**: Ensure your Pydantic schema matches function signature:
```python
# Schema and function must match
class MyInput(BaseModel):
    value: int  # Must match parameter type

async def my_fn(value: int) -> str:  # Types must align
    return str(value)
```

## Next Steps

- Review [Writing Custom Functions](./functions.md) for details that also apply to functions inside groups (type safety, streaming vs. single outputs, converters)
