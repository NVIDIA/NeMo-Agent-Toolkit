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


<!-- This role is needed at the index to set the default backtick role -->
```{eval-rst}
.. role:: py(code)
   :language: python
   :class: highlight
```

![NVIDIA Agent Intelligence Toolkit](./_static/aiqtoolkit_banner.png "AIQ Toolkit banner image")

# Welcome to the NVIDIA Agent Intelligence Toolkit Documentation

Agent Intelligence Toolkit (AIQ Toolkit) is a flexible library designed to seamlessly integrate your enterprise agents—regardless of framework—with various data sources and tools. By treating agents, tools, and agentic workflows as simple function calls, AIQ Toolkit enables true composability: build once and reuse anywhere.

## Key Features

- [**Framework Agnostic:**](./concepts/plugins.md) Works with any agentic framework, so you can use your current technology stack without replatforming.
- [**Reusability:**](./guides/sharing-workflows-and-tools.md) Every agent, tool, or workflow can be combined and repurposed, allowing developers to leverage existing work in new scenarios.
- [**Rapid Development:**](./guides/create-customize-workflows.md) Start with a pre-built agent, tool, or workflow, and customize it to your needs.
- [**Profiling:**](./guides/profiler.md) Profile entire workflows down to the tool and agent level, track input/output tokens and timings, and identify bottlenecks.
- [**Observability:**](./guides/observe-workflow-with-phoenix.md) Monitor and debug your workflows with any OpenTelemetry-compatible observability tool.
- [**Evaluation System:**](./guides/evaluate.md) Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [**User Interface:**](./guides/using-aiqtoolkit-ui-and-server.md) Use the AIQ Toolkit UI chat interface to interact with your agents, visualize output, and debug workflows.
- [**MCP Compatibility**](./components/mcp.md) Compatible with Model Context Protocol (MCP), allowing tools served by MCP Servers to be used as AIQ Toolkit functions.

With AIQ Toolkit, you can move quickly, experiment freely, and ensure reliability across all your agent-driven projects.

## Links

To learn more about AIQ Toolkit, see the following links:

* [About AIQ Toolkit](./intro/why-aiqtoolkit.md)
* [Install AIQ Toolkit](./intro/install.md)
* [Get Started](./intro/get-started.md)
* [Create and Customize Workflows](./guides/create-customize-workflows.md)
* [Sharing Components](./guides/sharing-workflows-and-tools.md)
* [Evaluating Workflows](./guides/evaluate.md)
* [Profiling a Workflow](./guides/profiler.md)
* [Observing a Workflow with Phoenix](./guides/observe-workflow-with-phoenix.md)
* [Command Line Interface](./concepts/cli.md)

## Feedback

We would love to hear from you! Please file an issue on [GitHub](https://github.com/NVIDIA/AIQToolkit/issues) if you have any feedback or feature requests.


```{toctree}
:maxdepth: 3

Introduction <./intro/index.md>
Guides <./guides/index.md>
Concepts <./concepts/index.md>
Components <./components/index.md>
Advanced <./advanced/index.md>
References <./references/index.md>
Troubleshooting <./troubleshooting.md>
Release Notes <./release-notes.md>
Code of Conduct <./code-of-conduct.md>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
