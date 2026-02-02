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
This section contains the release notes for [NeMo Agent toolkit](./index.md).

## Release 1.4.0
### Summary
This release introduces initial support for several frameworks and integrations including A2A, AWS Strands, Amazon Bedrock AgentCore, Microsoft Autogen, and NVIDIA Dynamo. In addition to new framework and integrations, an automatic agent wrapper for LangGraph Agents enables users to bring their own agent. Per-user functions enable deferred instantiation which provides per-user stateful functions, per-user resources, and other useful features. The toolkit continues to offer backwards compatibility, making the transition seamless for existing users.

### 🚀 Notable Features and Improvements
- [**A2A Support**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/components/integrations/a2a.md) NeMo Agent Toolkit now supports deploying and consuming agents using the A2A protocol.
- [**Amazon Bedrock AgentCore and Strands Agents Support:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/components/integrations/frameworks.md#strands) NeMo Agent Toolkit now supports building agents using Strands Agents framework and deploying them securely on Amazon Bedrock AgentCore runtime.
- [**LangChain Agent Automatic Wrapper:**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/examples/frameworks/auto_wrapper/langchain_deep_research/README.md) NeMo Agent Toolkit now supports automatic wrapping of existing LangChain/LangGraph Agents.
- [**Microsoft AutoGen Support**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/components/integrations/frameworks.md#autogen) NeMo Agent Toolkit now supports building agents using AutoGen framework.
- [**Initial NVIDIA Dynamo Integration:**](https://docs.nvidia.com/dynamo/latest/) NeMo Agent Toolkit now has initial Dynamo support for end-to-end deployment acceleration of agentic workflows.
- [**Per-User Functions**](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/docs/source/extend/custom-components/custom-functions/per-user-functions.md) NeMo Agent Toolkit now supports per-user functions for deferred instantiation, enabling per-user stateful functions, per-user resources, and other features.

Refer to the [changelog](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/CHANGELOG.md) for a complete list of changes.

## Known Issues
- Refer to [https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues](https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues) for an up to date list of current issues.
