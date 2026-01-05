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

# Memory in NVIDIA NeMo Agent Toolkit

The NeMo Agent toolkit Memory subsystem is designed to store and retrieve a user's conversation history, preferences, and other "long-term memory." This is especially useful for building stateful [LLM-based](./llms/index.md) applications that recall user-specific data or interactions across multiple steps.

The memory module is designed to be extensible, allowing developers to create custom memory back-ends, providers in NeMo Agent toolkit terminology.

## Included Memory Modules
The NeMo Agent toolkit includes three memory module providers, all of which are available as plugins:
* [Mem0](https://mem0.ai/) which is provided by the [`nvidia-nat-mem0ai`](https://pypi.org/project/nvidia-nat-mem0ai/) plugin.
* [Redis](https://redis.io/) which is provided by the [`nvidia-nat-redis`](https://pypi.org/project/nvidia-nat-redis/) plugin.
* [Zep](https://www.getzep.com/) which is provided by the [`nvidia-nat-zep-cloud`](https://pypi.org/project/nvidia-nat-zep-cloud/) plugin.

## Examples
The following examples in the [repository](https://github.com/NVIDIA/NeMo-Agent-Toolkit) demonstrate how to use the memory module in the NeMo Agent toolkit:
* `examples/memory/redis`
* `examples/frameworks/semantic_kernel_demo`
* `examples/RAG/simple_rag`

## Additional Resources
For information on how to write a new memory module provider can be found in the [Adding a Memory Provider](../extend/custom-components/memory.md) document.
