# Creating Custom Tools and Function Groups

How to register individual tools (`register_function`, `FunctionInfo.from_fn`, the `Function` class) and function groups (`register_function_group`). Companion: [`../../nat-evaluation/references/code-patterns.md`](../../nat-evaluation/references/code-patterns.md) has copy-paste-ready code templates.

NeMo Agent Toolkit provides two mechanisms for registering tools: **individual functions** for standalone tools, and **function groups** for related tools that share configuration or resources.

## Individual Tool (register_function)

Define the tool in a separate module and **import it before calling `WorkflowBuilder.from_config()`**. The `@register_function` decorator registers it with the global type registry at import time, so the builder can resolve it from the YAML config.

### Simple pattern using `FunctionInfo.from_fn()` (recommended)

This is the pattern used by most production NeMo Agent Toolkit projects. Yield a `FunctionInfo` wrapping a plain async function:

```python
# tools/chitchat_tool.py
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef
from pydantic import Field


class ChitchatToolConfig(FunctionBaseConfig, name="chitchat_tool"):
    description: str = Field(default="General conversation and fallback tool")
    llm_name: LLMRef  # typed reference to an LLM defined in the llms: section


@register_function(
    config_type=ChitchatToolConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],  # required when using LangChain LLMs
)
async def chitchat_tool(config: ChitchatToolConfig, builder: Builder):
    from langchain_core.prompts import PromptTemplate

    # Get the LLM from the builder using the config reference
    llm = await builder.get_llm(
        llm_name=config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    prompt = PromptTemplate(input_variables=["query"], template="Answer: {query}")
    chain = prompt | llm

    async def arun(message: str) -> str:
        response = await chain.ainvoke(message)
        return str(response.content)

    yield FunctionInfo.from_fn(arun, description=config.description)
```

**Key patterns:**

- **`LLMRef`**: Use `from nat.data_models.component_ref import LLMRef` for config fields that reference an LLM by name. In the YAML, this maps to `llm_name: my_llm`.
- **`framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]`**: Required on `@register_function` when the tool uses LangChain LLMs via `builder.get_llm()`.
- **`builder.get_llm()`**: Access configured LLMs inside tools. Pass `wrapper_type=LLMFrameworkEnum.LANGCHAIN` to get a LangChain `BaseChatModel`.
- **`FunctionInfo.from_fn()`**: Wraps a plain `async def` as a NeMo Agent Toolkit function. Simpler than the `Function` class.
- **Lazy imports**: Import heavy dependencies (LangChain, httpx) inside the function body, not at module top level.

### Advanced pattern using `Function` class

For tools that need both single-output and streaming modes, use the `Function` class with typed input/output schemas:

```python
# tools/my_tool.py
from pydantic import BaseModel
from collections.abc import AsyncGenerator
from nat.data_models.function import FunctionBaseConfig
from nat.builder.function import Function
from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function


class MyInput(BaseModel):
    query: str

class MyOutput(BaseModel):
    answer: str

class MyStreamingOutput(BaseModel):
    result: str

class MyToolConfig(FunctionBaseConfig, name="my_tool"):
    pass


@register_function(config_type=MyToolConfig)
async def my_tool(config: MyToolConfig, builder: Builder):
    class MyTool(Function[MyInput, MyStreamingOutput, MyOutput]):
        def __init__(self, config: MyToolConfig):
            super().__init__(config=config, description="My custom tool")

        async def _ainvoke(self, value: MyInput) -> MyOutput:
            return MyOutput(answer=f"Result for: {value.query}")

        async def _astream(self, value: MyInput) -> AsyncGenerator[MyStreamingOutput, None]:
            yield MyStreamingOutput(result=f"Result for: {value.query}")

    yield MyTool(config=config)
```

### Triggering registration

Import the tool module before building the workflow. Two options:

**Option A — explicit import in main.py:**

```python
import tools.my_tool  # noqa: F401 — import triggers @register_function
```

**Option B — entry point in pyproject.toml (recommended for packages):**

```toml
[project.entry-points."nat.components"]
register = "my_package.register"
```

Then in `my_package/register.py`, import all tool modules:

```python
from .tools import chitchat_tool  # noqa: F401
from .tools import search_tool  # noqa: F401
```

The entry point only registers once the project is installed in the venv. Make sure `pyproject.toml` has a `[build-system]` block:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

Without it, `uv sync` skips installing the local project, the entry point is inert, and a fresh subprocess (e.g. `nat eval`) won't find your custom `_type` even though the decorator is in your source.

NeMo Agent Toolkit auto-discovers entry points at startup — no manual imports needed.

Reference the tool in `workflow.yml` by its registered `_type`:

```yaml
functions:
  my_tool:
    _type: my_tool  # matches name= in MyToolConfig
    llm_name: nim_llm  # if the tool has an LLMRef field
```

## Function Groups (register_function_group)

When you have multiple related tools that share configuration or resources (e.g., a database connection, an API client, an LLM reference), use a function group. A function group registers multiple tools under a single config block, and each tool is accessible as `<group_name>__<tool_name>`.

```python
# tools/data_tools.py
from collections.abc import AsyncGenerator
from pydantic import BaseModel, Field
from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig


class QueryInput(BaseModel):
    """Pydantic model for typed tool input."""
    sql: str
    limit: int = 100


class DataToolsConfig(FunctionGroupBaseConfig, name="data_tools"):
    """Configuration shared by all data tools in this group."""
    database_url: str = Field(description="Connection string for the database")
    llm_name: str = Field(default="nim_llm", description="LLM for natural language queries")


@register_function_group(
    config_type=DataToolsConfig,
    framework_wrappers=[LLMFrameworkEnum.LANGCHAIN],
)
async def data_tools(config: DataToolsConfig, builder: Builder) -> AsyncGenerator[FunctionGroup, None]:
    # Create shared resources once
    import httpx
    db = await connect_to_database(config.database_url)
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    group = FunctionGroup(config=config)

    # Simple tools — plain async functions with string I/O
    async def list_tables(input_message: str) -> str:
        tables = await db.list_tables()
        return ", ".join(tables)

    # Typed tools — use input_schema for Pydantic model input
    async def query_table(input_data: QueryInput) -> str:
        results = await db.query(input_data.sql, limit=input_data.limit)
        return str(results)

    async def describe_schema(input_message: str) -> str:
        schema = await db.describe(input_message)
        return str(schema)

    group.add_function("query", query_table, input_schema=QueryInput, description="Execute a database query")
    group.add_function("list_tables", list_tables, description="List all available tables")
    group.add_function("describe_schema", describe_schema, description="Describe the schema of a table")

    yield group
```

Reference the function group in `workflow.yml`:

```yaml
function_groups:
  my_data_tools:
    _type: data_tools
    database_url: "postgresql://localhost/mydb"
    llm_name: nim_llm

workflow:
  _type: react_agent
  tool_names: [my_data_tools]   # reference the group — all its tools become available
  llm_name: nim_llm
  verbose: true
```

The agent sees three tools: `my_data_tools__query`, `my_data_tools__list_tables`, `my_data_tools__describe_schema`. You can also control which tools from the group are exposed:

```yaml
function_groups:
  my_data_tools:
    _type: data_tools
    database_url: "postgresql://localhost/mydb"
    include: [query, list_tables]     # only expose these two tools
    # OR
    exclude: [describe_schema]        # expose all except this one
```

When using function groups in a `sequential_executor`, reference individual tools with dot notation:

```yaml
workflow:
  _type: sequential_executor
  tool_list: [my_data_tools.query, my_data_tools.describe_schema]
```
