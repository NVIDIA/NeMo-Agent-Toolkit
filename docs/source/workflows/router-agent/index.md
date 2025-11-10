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

# About the Router Agent

The router agent is a control flow component that analyzes incoming requests and directs them to the most appropriate branch based on the request configuration. The agent pairs single-pass architecture with intelligent request routing to analyze prompts and selects one branch that best handles the request. The agent is ideal for scenarios where different types of requests need specialized handling.

The agent uses NVIDIA NeMo Agent toolkit's core library agents and tools to simplify your development experience and deployment. Additionally, you can customize prompts in your YAML config files for your specific needs.

## High-Level Breakdown of the Router Agent

The router agent's implementation uses a two-node graph structure:
1. **Router Node**: In the routing phase, analyzes the request and selects the appropriate branch.
2. **Branch Node**: In the execution phase, executes the selected branch and returns the result.