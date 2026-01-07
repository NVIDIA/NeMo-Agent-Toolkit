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

# About Automatic Memory Wrapper Agent

The `auto_memory_agent` wraps any NeMo Agent Toolkit agent to provide **automatic memory capture and retrieval** without requiring the LLM to invoke memory tools explicitly. Unlike traditional tool-based memory where LLMs may forget to call memory tools, this wrapper guarantees consistent memory operations on every interaction while maintaining full compatibility with any agent type (ReAct, ReWOO, Tool Calling, Reasoning, etc.).

The agent uses the NVIDIA NeMo Agent toolkit core library to simplify development. Additionally, you can customize behavior through YAML config options for your specific needs.

To configure your automatic memory wrapper agent, refer to [Configure the Automatic Memory Wrapper Agent](./auto-memory-wrapper.md).

```{toctree}
:hidden:
:caption: Automatic Memory Wrapper

Configure Automatic Memory Wrapper Agent <./auto-memory-wrapper.md>
```
