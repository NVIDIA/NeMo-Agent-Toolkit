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

# About Improving NVIDIA NeMo Agent Toolkit Workflows

NeMo Agent toolkit offers a variety of [tools](../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) and techniques to improve [workflows](../build-workflows/about-building-workflows.md). This section provides guides on evaluating, profiling, optimizing, and scaling your workflows for better performance and efficiency.

- [Evaluating Workflows](./evaluate.md) - Validate and maintain accuracy of agentic workflows with built-in evaluation tools.
- [Profiling and Performance Monitoring](./profiler.md) - Use the profiler to profile entire workflows down to the [tool](../build-workflows/functions-and-function-groups/functions.md#agents-and-tools) and [agent](../components/agents/index.md) level, track input/output tokens and timings, and identify bottlenecks.
- [Optimizer Guide](./optimizer.md) - Automatically tune the parameters and prompts of your agents, tools, and workflows to maximize performance, minimize cost, and increase accuracy.
- [Sizing Calculator](./sizing-calc.md) - Using the sizing calculator to estimate GPU cluster size requirements.
- [Test Time Compute](./test-time-compute.md) - Use composable pre-built or customizable strategies to scale agent execution at runtime and improve performance.
- [Finetuning Harness](./finetuning/index.md) - Leverage the finetuning harness for finetuning of agentic [LLM](../build-workflows/llms/index.md) workflows to iteratively improve agents through experience.