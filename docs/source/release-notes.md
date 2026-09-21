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
* Add `HITLMiddleware` for human-in-the-loop function interception
* Enable preflight authentication for applicable authentication providers
* Track LangChain Runnable callbacks
* Add MLflow OTLP telemetry exporter, docs, and example
* Export runtime context and interactive HITL models
* Add opt-in provider hooks for generated ids and timestamps
* Export the interactive prompt content models
* Route interaction prompt ids and timestamps via providers
* Add CircuitBreakerMiddleware for tool fault tolerance
* Remove `local_sandbox`
* Don't expose the `user_id` parameter to the LLM
* Improved user identity resolution

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.9/CHANGELOG.md) for the complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
