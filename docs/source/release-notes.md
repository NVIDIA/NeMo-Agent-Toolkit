<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# NVIDIA NeMo Agent Toolkit Release Notes
This section contains the release notes for [NeMo Agent Toolkit](./index.md).

## Release 1.5.0
### Summary
This release expands runtime intelligence, framework-level performance acceleration, and production observability in the toolkit, while making workflow publishing to MCP ecosystems easier.

**Migration notice:** Release `1.5.0` includes packaging and compatibility refactors (including meta-package changes, eval package split, and import-path updates). Review the [Migration Guide](./resources/migration-guide.md#v150) before upgrading.

- [**Dynamo Runtime Intelligence:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/examples/dynamo_integration/latency_sensitivity_demo/README.md) Automatically infer per-request latency sensitivity from agent profiles and apply runtime hints for cache control, load-aware routing, and priority-aware serving.
- [**Agent Performance Primitives (APP):**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/packages/nvidia_nat_app/src/meta/pypi.md) Introduce framework-agnostic performance primitives that accelerate graph-based agent frameworks such as LangChain, CrewAI, and Agno with parallel execution, speculative branching, and node-level priority routing.
- [**LangSmith Native Integration:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/docs/source/run-workflows/observe/observe-workflow-with-langsmith.md) Observe end-to-end agent execution with native LangSmith tracing, run evaluation experiments, compare outcomes, and manage prompt versions across development and production workflows.
- [**FastMCP Workflow Publishing:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/docs/source/run-workflows/fastmcp-server.md) Publish NeMo Agent Toolkit workflows as MCP servers using the FastMCP runtime to simplify MCP-native deployment and integration.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.5/CHANGELOG.md) for the complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
