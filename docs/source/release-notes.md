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

## Release v1.9.0
### Summary
* feat(middleware): add HITLMiddleware for human-in-the-loop function interception by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2060
* Enable preflight authentication for applicable authentication providers by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2078
* Track LangChain Runnable callbacks by @WilliamK112 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2100
* Add MLflow OTLP telemetry exporter, docs, and example by @EnesYilmazcode in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2112
* feat(plugin-api): export runtime context and interactive HITL models by @DABH in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2113
* feat(core): add opt-in provider hooks for generated ids and timestamps by @DABH in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2114
* feat(plugin-api): export the interactive prompt content models by @DABH in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2145
* feat(core): route interaction prompt ids and timestamps via providers by @DABH in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2144
* feat(core): Add CircuitBreakerMiddleware for tool fault tolerance by @sankhyanreyansh in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2158
* chore: Remove `local_sandbox` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2194
* fix: Don't expose the `user_id` parameter to the LLM by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2190
* Improved user identity resolution by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/2197