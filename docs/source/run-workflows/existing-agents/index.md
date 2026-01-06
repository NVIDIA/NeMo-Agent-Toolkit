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

# Running Existing Agents in NVIDIA NeMo Agent Toolkit

NeMo Agent toolkit provides automatic wrapper functionality that allows you to integrate existing agents from other frameworks without rewriting them. This enables you to leverage features such as observability, evaluation, and configuration management while continuing to use your existing agent implementations.

This approach is particularly valuable for users who are just getting started with NeMo Agent Toolkit. You can begin taking advantage of features offered by the toolkit right away with your existing agents, then gradually adopt more native features as you become familiar with the platform.

## How Automatic Wrappers Work

NeMo Agent toolkit is a library that wraps around existing frameworks to add instrumentation and utilizes that instrumentation to provide advanced features such as observability, evaluation, and configuration management.

### Accessing the Builder

Before version 1.4, users needed to add wrappers around their code to access the `Builder` class. Starting with version 1.4, the `Builder` class can be accessed at any time using the `Builder.current()` and `SyncBuilder.current()` functions, avoiding the need to wrap classes.

While this new approach simplifies integration, wrapping agents is still the preferred method. Wrapping enforces good design patterns, promotes resource reuse, and improves performance. However, both methods should yield identical results.

## Supported Frameworks

NeMo Agent toolkit currently provides automatic wrappers for the following frameworks:

- [LangGraph](langgraph.md): Integrate existing LangGraph agents and workflows

## Benefits of Using Automatic Wrappers

The automatic wrapper approach provides several advantages:

- **No Rewrite Required**: Run existing agents without modifying their core implementation
- **Unified Configuration**: Use the NeMo Agent toolkit YAML configuration system to manage agents
- **Observability**: Add tracing and monitoring through supported observability platforms
- **Evaluation**: Leverage the toolkit evaluation framework to measure agent performance
- **LLM Flexibility**: Easily swap between different LLMs through configuration
- **Deployment Options**: Use deployment capabilities such as MCP server, A2A server, and REST API provided by the toolkit

## When to Use Automatic Wrappers

Automatic wrappers are ideal when:

- You have existing agent implementations that work well
- You want to quickly add NeMo Agent toolkit capabilities without refactoring
- You need to evaluate multiple frameworks and compare performance
- You want to leverage the deployment and serving features provided by the toolkit

## Limitations

While automatic wrappers provide significant benefits, there are some limitations:

- **Framework-Specific Features**: Some framework-specific features may not be fully supported
- **Configuration Constraints**: Custom configuration classes cannot be used out of the box with automatic wrappers, but can be used with code modifications
- **Code Modifications**: Minor code changes may be required to make agents configurable
- **State Management**: Complex state management patterns may need adaptation
- **Threading**: Automatic wrappers are not thread-safe and should not be used in multi-threaded environments.

For detailed information on framework-specific limitations, refer to the individual framework documentation pages.

## Getting Started

To get started with automatic wrappers:

1. Choose the framework you want to integrate from the list above
2. Follow the framework-specific guide to set up your configuration
3. Make any necessary code modifications as described in the guide
4. Run your agent using standard NeMo Agent toolkit commands

Each framework guide provides complete examples and step-by-step instructions for integration.

```{toctree}
:titlesonly:

LangGraph <./langgraph.md>
```
