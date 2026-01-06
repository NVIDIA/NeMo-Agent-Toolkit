<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Running Existing LangGraph Agents in NVIDIA NeMo Agent Toolkit

NVIDIA NeMo Agent toolkit provides a `langgraph_wrapper` workflow type that allows you to integrate existing LangGraph agents with minimal changes to your code. This wrapper enables you to run LangGraph agents through the toolkit while adding configuration management, observability, and evaluation capabilities.

## Prerequisites

Ensure you have installed the required packages:

```bash
uv pip install nvidia-nat-langchain
```

## Basic Configuration

The `langgraph_wrapper` workflow type requires a minimal configuration file that points to your LangGraph agent implementation.

### Configuration Example

<!-- path-check-skip-begin -->
```yaml
workflow:
  _type: langgraph_wrapper
  dependencies:
    - path/to/your/agent/package
  graph: path/to/agent.py:agent
  env: .env
```
<!-- path-check-skip-end -->

### Configuration Parameters

The following table describes the configuration parameters for the `langgraph_wrapper`:

| Parameter | Type | Description | Required |
|-----|----|----|---|
| `_type` | string | Must be set to `langgraph_wrapper` | Yes |
| `dependencies` | list[string] | List of directories paths to add to the python path | No |
| `graph` | string | Path to the graph definition in the format `module_path:variable_name` | Yes |
| `env` | string or dict | Path to `.env` file or dictionary of environment variables | No |
| `description` | string | Description of the workflow | No |

The configuration parameters mirror the [LangGraph CLI configuration file](https://docs.langchain.com/langsmith/cli#configuration-file), enabling compatibility with existing LangGraph deployments.

## Running Without Code Changes

For simple LangGraph agents, you can run them directly through the wrapper without any code modifications:

<!-- path-check-skip-begin -->
```yaml
workflow:
  _type: langgraph_wrapper
  dependencies:
    - external/my-langgraph-agent
  graph: external/my-langgraph-agent/agent.py:agent
  env: .env
```
<!-- path-check-skip-end -->

This configuration works when your agent:

- Uses hardcoded LLM configurations
- Does not require dynamic configuration
- Has all necessary environment variables in the `.env` file

You can then run the agent using standard NeMo Agent toolkit commands:

```bash
nat run --config_file config.yml --input "What is LangSmith?"
```

## Making Agents Configurable

To make your LangGraph agent configurable through the NeMo Agent toolkit configuration system, you need to modify your agent code to retrieve LLMs, Embeddings, Tools, etc. from the NeMo Agent toolkit builder.

### When Code Changes Are Necessary

You should modify your agent code when you want to:

- Use different components (LLMs, tools, embedders, object stores) through configuration without changing code
- Leverage the configuration management features provided by the toolkit
- Make the agent configurable for different environments
- Enable easy component switching for testing and evaluation

### Modifying Your Agent Code

To make your agent configurable, replace hardcoded component initialization with calls to the NeMo Agent toolkit builder. The following example demonstrates this pattern using LLMs, but the same approach works for tools, embedders, object stores, and other components.

#### Original Agent Code

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from deepagents import create_deep_agent

# Hardcoded model initialization
model = ChatGoogleGenerativeAI(model="gemini-3-pro-preview", temperature=0.0)

# Create the agent
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
)
```

#### Modified Agent Code

```python
from deepagents import create_deep_agent
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.sync_builder import SyncBuilder

# Get model from NeMo Agent toolkit configuration
model = SyncBuilder.current().get_llm("agent", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

# Create the agent
agent = create_deep_agent(
    model=model,
    tools=[tavily_search, think_tool],
    system_prompt=INSTRUCTIONS,
)
```

This single-line change enables you to configure the LLM through your YAML configuration file.

### Configuring Other Components

The same pattern applies to other components. Here are examples:

#### Tools

```python
# Get tools from configuration
tools = SyncBuilder.current().get_tools(["search_tool", "calculator_tool"], wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

#### Embedders

```python
# Get embedder from configuration
embedder = SyncBuilder.current().get_embedder("text_embedder", wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

#### Object Stores

```python
# Get object store from configuration
object_store = SyncBuilder.current().get_object_store_client("vector_store")
```

For more information on available builder methods, refer to the [Building Workflows Documentation](../../build-workflows/about-building-workflows.md).

### Updated Configuration

With the modified agent code, you can now specify components in your configuration:

<!-- path-check-skip-begin -->
```yaml
llms:
  agent:
    _type: openai
    model: azure/openai/gpt-4
    base_url: https://integrate.api.nvidia.com/v1
    api_key: ${NVIDIA_API_KEY}

tools:
  search_tool:
    _type: tavily_search
    api_key: ${TAVILY_API_KEY}

embedders:
  text_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5

workflow:
  _type: langgraph_wrapper
  dependencies:
    - external/my-langgraph-agent
  graph: path/to/configurable_agent.py:agent
  env: .env
```
<!-- path-check-skip-end -->

Now you can change components by modifying the configuration without touching your agent code:

<!-- path-check-skip-begin -->
```yaml
llms:
  agent:
    _type: openai
    model: gcp/google/gemini-3-pro
    api_key: ${NVIDIA_API_KEY}
```
<!-- path-check-skip-end -->

## Adding Observability

You can add observability to your LangGraph agent by including telemetry configuration:

<!-- path-check-skip-begin -->
```yaml
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: my-langgraph-agent

llms:
  agent:
    _type: openai
    model: azure/openai/gpt-4
    base_url: https://integrate.api.nvidia.com/v1
    api_key: ${NVIDIA_API_KEY}

workflow:
  _type: langgraph_wrapper
  dependencies:
    - external/my-langgraph-agent
  graph: path/to/configurable_agent.py:agent
  env: .env
```
<!-- path-check-skip-end -->

For more information on observability options, refer to the [Observability Documentation](../observe/observe.md).

## Building Workflows

For information on how to build and structure your workflows, including configuration options and best practices, refer to the [Building Workflows Documentation](../../build-workflows/about-building-workflows.md).

## Limitations and Considerations

### Graph Definition Requirements

The graph definition specified in the configuration must be either:

- A `CompiledStateGraph` instance
- A callable that returns a `CompiledStateGraph` when invoked with a `RunnableConfig`

Other LangGraph graph types may not be supported.

### Message Format

The wrapper expects input in message format compatible with LangChain's message types. The wrapper automatically converts single inputs to message format, but complex input structures may require additional handling.

### State Management

LangGraph agents with complex state management patterns may need additional configuration or code modifications to work correctly with the wrapper.

### Environment Variables

Environment variables specified in the `env` parameter are loaded before the graph is initialized. Ensure all required variables are available in the specified environment file or system environment.

### Dependency Loading

Dependencies listed in the `dependencies` parameter are added to the Python path before loading the graph. Ensure these paths are accessible from your execution environment.

## Complete Example

For a comprehensive example of integrating a LangGraph agent, see the Deep Research agent example in the repository:

- **Location**: `examples/frameworks/auto_wrapper/langchain_deep_research/`
- **Notebook**: `langgraph_deep_research.ipynb`

This example demonstrates:

- Running an existing LangGraph agent without code changes
- Making agents configurable with different components (LLMs, tools, embedders)
- Adding Phoenix telemetry for observability
- Evaluating agent performance with automated metrics

## Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Building Workflows](../../build-workflows/about-building-workflows.md)
- [Workflow Configuration](../../build-workflows/workflow-configuration.md)
- [Observability](../observe/observe.md)
- [Evaluation](../../improve-workflows/evaluate.md)
