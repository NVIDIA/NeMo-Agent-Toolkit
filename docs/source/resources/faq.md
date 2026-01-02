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

# NVIDIA NeMo Agent Toolkit FAQs
NVIDIA NeMo Agent toolkit frequently asked questions (FAQs).

## Do I Need to Rewrite All of my Existing Code to Use NeMo Agent Toolkit?
No, NeMo Agent toolkit is **100% opt in.** While we encourage users to wrap (decorate) every [tool](../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) and [agent](../components/agents/index.md) to get the most out of the [profiler](../improve-workflows/profiler.md), you have the freedom to integrate to whatever level you want - tool level, agent level, or entire [workflow](../build-workflows/about-building-workflows.md) level. You have the freedom to start small and where you believe you will see the most value and expand from there.

## Is NeMo Agent Toolkit another LLM or Agentic Framework?
No, NeMo Agent toolkit is designed to work alongside, not replace, your existing agentic frameworks — whether they are enterprise-grade systems or simple Python-based agents.

## Is NeMo Agent Toolkit An Attempt to Solve Agent-to-Agent Communication?
No, agent communication is best handled over existing protocols, such as MCP, HTTP, gRPC, and sockets.

## Is NeMo Agent Toolkit an Observability Platform?
No, while NeMo Agent toolkit is able to collect and transmit fine-grained telemetry to help with optimization and [evaluation](../improve-workflows/evaluate.md), it does not replace your preferred observability platform and data collection application.
