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
# About the Tool Calling Agent
A tool calling agent is an AI system that directly invokes external tools based on structured function definitions. Unlike ReAct agents, it does not reason between steps but instead selects tools based on predefined function schemas. The agent examines the tool name, description, and input parameter schema to determine which tool to invoke. Some LLMs do not support tool or function calling or being used with tool calling agents.

The agent uses the NVIDIA NeMo Agent toolkit core library agents and tools to simplify your development experience and deployment. Additionally, you can customize prompts in your YAML config files for your specific needs.