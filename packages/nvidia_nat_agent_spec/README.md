# NVIDIA NeMo Agent Toolkit - Agent-Spec Plugin

This plugin integrates Oracle Agent-Spec with NeMo Agent Toolkit, allowing you to run Agent-Spec YAML configurations as NAT workflows with full observability, profiling, and evaluation capabilities.

## Overview

The Agent-Spec plugin converts Oracle Agent-Spec YAML configurations into executable NAT Functions by:

1. Loading Agent-Spec YAML files (Oracle's framework-agnostic agent specification format)
2. Converting them to LangGraph `CompiledStateGraph` using `pyagentspec`'s runtime adapters
3. Wrapping the graph as a NAT Function that can be used in NAT workflows
4. Providing observability, profiling, and other NAT features

## Installation

### From PyPI

Install the plugin as part of NeMo Agent Toolkit:

```bash
uv pip install "nvidia-nat[agent-spec]"
```

### From Source (Workspace)

If working with the NeMo Agent Toolkit workspace:

```bash
cd /path/to/NeMo-Agent-Toolkit
uv sync --extra agent-spec
```

**Note**: The Agent-Spec plugin conflicts with LangChain-related extras (`langchain`, `vanna`, `most`). You cannot sync with both `agent-spec` and these extras simultaneously. See [Known Limitations](#known-limitations) below.

### Standalone Installation

Or install the plugin package directly:

```bash
cd packages/nvidia_nat_agent_spec
uv pip install -e .
```

## Known Limitations

### Dependency Conflict with LangChain-Related Plugins

**Important**: The Agent-Spec plugin cannot be used simultaneously with LangChain-related plugins in the same workspace due to incompatible dependency versions:

- **`pyagentspec`** (used by agent-spec) requires: `langchain-core<1.0.0`, `langgraph<1.0.0`
- **`nvidia-nat-langchain`** requires: `langchain-core>=1.2.6,<2.0.0`, `langgraph>=1.0.5,<2.0.0`

These version ranges are incompatible and cannot coexist in the same workspace.

**Affected extras**: The following NAT extras conflict with `agent-spec`:
- `langchain` - Direct LangChain plugin
- `vanna` - Depends on LangChain plugin
- `most` - Includes LangChain plugin

#### Workarounds

1. **Use separate environments**: Install Agent-Spec in a separate Python environment when you need it
2. **Choose one plugin**: Use either Agent-Spec or LangChain-related plugins, but not both in the same workspace
3. **Sync with only agent-spec**: When syncing the workspace, use `uv sync --extra agent-spec` without conflicting extras

The root `pyproject.toml` includes conflict declarations that prevent installing incompatible extras together. `uv sync` will fail if you attempt to sync with both `agent-spec` and conflicting extras.

## Usage

### Basic Example

Create a NAT workflow configuration file (`example_agent_spec_config.yml`):

```yaml
general:
  telemetry:
    logging:
      console:
        _type: console
        level: INFO
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: agent-spec-example

workflow:
  _type: agent_spec_wrapper
  spec_file: path/to/your/agent_spec.yaml
  description: "Agent-Spec agent wrapped as NAT Function"
```

Run the workflow:

```bash
nat run --config_file example_agent_spec_config.yml --input "Hello!"
```

### Configuration Options

The `agent_spec_wrapper` configuration supports the following options:

- **`spec_file`** (required): Path to the Agent-Spec YAML configuration file
- **`description`** (optional): Description of the workflow
- **`tool_registry`** (optional): Dictionary mapping tool names to LangGraph tools or callables
- **`components_registry`** (optional): Dictionary mapping component IDs to LangGraph components (e.g., LLMs). Can be used to override components defined in the Agent-Spec YAML
- **`checkpointer`** (optional): LangGraph checkpointer (e.g., `MemorySaver`). Required when using `ClientTool` in Agent-Spec YAML. If not provided and `ClientTool` is detected, `MemorySaver` will be used automatically

### Example with Tool Registry

```yaml
workflow:
  _type: agent_spec_wrapper
  spec_file: agent_with_tools.yaml
  description: "Agent with custom tools"
  tool_registry:
    get_weather: |
      def get_weather(city: str) -> str:
          return f"The weather in {city} is sunny."
```

### Example with Components Override

```yaml
workflow:
  _type: agent_spec_wrapper
  spec_file: agent.yaml
  description: "Agent with NAT LLM override"
  components_registry:
    llm_id: |
      # Your NAT LLM configuration here
```

## Features

- **Automatic Checkpointer Detection**: Automatically detects `ClientTool` usage in Agent-Spec YAML and creates a `MemorySaver` checkpointer if needed
- **Tool Registry Support**: Pass custom tools to Agent-Spec workflows
- **Component Override**: Override components (such as LLMs) defined in Agent-Spec YAML with NAT components
- **Full NAT Integration**: Benefit from all NAT features including evaluation, profiling, observability (Phoenix), and middleware

## Requirements

- Python 3.11, 3.12, or 3.13
- `pyagentspec[langgraph,langgraph_mcp]>=26.1.0`
- NeMo Agent Toolkit Core (`nvidia-nat-core`)

## Testing

Run the test suite:

```bash
cd /path/to/NeMo-Agent-Toolkit

# Sync with agent-spec and test dependencies
uv sync --extra agent-spec --extra test

# Run all tests
uv run pytest packages/nvidia_nat_agent_spec/tests/ -v

# Or run specific tests
uv run pytest packages/nvidia_nat_agent_spec/tests/test_nim_integration.py::test_load_nim_agent_spec -v
```

**Note**:
- Some integration tests require `NVIDIA_API_KEY` environment variable and will be skipped if not set
- If `pytest` or `pytest-asyncio` are not available, install them: `uv pip install pytest pytest-asyncio`

## Documentation

For more information about:
- **Agent-Spec**: See [Oracle Agent-Spec documentation](https://github.com/oracle/agent-spec)
- **NeMo Agent Toolkit**: See [NeMo Agent Toolkit documentation](https://docs.nvidia.com/nemo/agent-toolkit/latest/)
- **Plugin Design**: See `DESIGN.md` in this directory

## License

Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
