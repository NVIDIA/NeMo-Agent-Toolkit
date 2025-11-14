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

# About ReAct Agent
This is a ReAct (Reasoning and Acting) agent, based on the [ReAct paper](https://react-lm.github.io/). The ReAct agent's prompt is directly inspired by the prompt examples in the appendix of the paper. The agent uses the NVIDIA NeMo Agent toolkit core library agents and tools to perform ReAct reasoning between tool calls. In your YAML config files, you can customize prompts for your specific needs. 

To configure your ReAct agent, refer to [Configure the ReAct Agent](./react-agent.md).