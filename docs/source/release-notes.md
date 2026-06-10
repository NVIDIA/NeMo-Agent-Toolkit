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

## Release v1.8.0
### Summary

* Migrated Tavily web search support from the LangChain-only `tavily_internet_search` tool in `nvidia-nat[langchain]` to the provider-managed, framework-agnostic [`nemo-agent-toolkit-tavily`](https://github.com/tavily-ai/NeMo-Agent-Toolkit-tavily) plugin. Workflows should install the Tavily package, configure Tavily under `function_groups` with `_type: tavily`, and reference exposed tools by names such as `internet_search__search`.
* Added new public plugin API documentation for externally managed NeMo Agent Toolkit plugin packages. Thank you to the maintainers of [`NeMo-Agent-Toolkit-Tavily`](https://github.com/tavily-ai/NeMo-Agent-Toolkit-tavily) and [`NeMo-Agent-Toolkit-ATR`](https://github.com/Agent-Threat-Rule/NeMo-Agent-Toolkit-atr) for building on the new public API surface. These third-party repositories are managed outside the NeMo Agent Toolkit repository and may release, change, or run CI on schedules independent of NeMo Agent Toolkit release cycles.

### Breaking Changes

* The `tavily_internet_search` function in `nvidia-nat[langchain]` has been removed as a working Tavily search implementation. Existing workflow configuration files that use `_type: tavily_internet_search` must install `nemo-agent-toolkit-tavily` and migrate to the Tavily function group. See the [migration guide](./resources/migration-guide.md#tavily-internet-search-package-migration-breaking) for details.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.8/CHANGELOG.md) for the complete list of changes.

## Release v1.7.0
### Summary

* Add AI coding agent skills for NeMo Agent Toolkit
* Add ATIF trajectory exporter for Phoenix visualization and debugging
* Add Exa Search API support as internet search tool
* Add OCI LangChain support for hosted Nemotron workflows
* ATOF v0.1: Agentic Trajectory Observability Format
* Add consent-gated runtime telemetry for NeMo Agent Toolkit CLI commands
* Add Arize AX OTLP exporter, docs, and examples
* Add token streaming support for ReAct Agent

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.7/CHANGELOG.md) for the complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
