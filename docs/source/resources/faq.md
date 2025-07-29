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

# NVIDIA NeMo Agent Toolkit FAQ
NVIDIA NeMo Agent toolkit frequently asked questions (FAQ).

## Do I Need to Rewrite All of my Existing Code to Use NeMo Agent Toolkit?
No, NeMo Agent toolkit is **100% opt in.** While we encourage users to wrap (decorate) every tool and agent to get the most out of the profiler, you have the freedom to integrate to whatever level you want - tool level, agent level, or entire workflow level. You have the freedom to start small and where you believe you’ll see the most value and expand from there.

## Is NeMo Agent Toolkit another LLM or Agentic Framework?
No, NeMo Agent toolkit is designed to work alongside, not replace, your existing agentic frameworks — whether they are enterprise-grade systems or simple Python-based agents.

## Is NeMo Agent Toolkit An Attempt to Solve Agent-to-Agent Communication?
No, agent communication is best handled over existing protocols, such as MCP, HTTP, gRPC, and sockets.

## Is NeMo Agent Toolkit an Observability Platform?
No, while NeMo Agent toolkit is able to collect and transmit fine-grained telemetry to help with optimization and evaluation, it does not replace your preferred observability platform and data collection application.
