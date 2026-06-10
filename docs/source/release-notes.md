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
* Migrated Redis memory and object store support out of the NeMo Agent Toolkit repository and into the Redis-managed [`nemo-agent-toolkit-redis`](https://pypi.org/project/nemo-agent-toolkit-redis/) plugin. Workflows that use Redis should install the Redis plugin package directly, or use the `nvidia-nat[redis]` extra which now resolves the third-party package.

### Breaking Changes

* The `tavily_internet_search` function in `nvidia-nat[langchain]` has been removed as a working Tavily search implementation. Existing workflow configuration files that use `_type: tavily_internet_search` must install `nemo-agent-toolkit-tavily` and migrate to the Tavily function group. See the [migration guide](./resources/migration-guide.md#tavily-internet-search-package-migration-breaking) for details.
* Redis memory and object store implementations are no longer shipped from the NeMo Agent Toolkit source tree. The historical `nvidia-nat-redis` distribution remains as a compatibility package, but new installs should use `nemo-agent-toolkit-redis` or `nvidia-nat[redis]`. See the [migration guide](./resources/migration-guide.md#redis-package-migration-breaking) for details.

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
