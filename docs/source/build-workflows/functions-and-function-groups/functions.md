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

# Functions

Functions are reusable components that perform specific operations, such as web searches, API calls, or calculations.

In NeMo Agent toolkit, functions are a core abstraction that offer type-safe, asynchronous operations with support for both single and streaming outputs. They wrap callable objects (like Python functions or coroutines) and enhance them with:

* Type validation and conversion
* Schema-based input/output validation via Pydantic models
* Unified interfaces to improve composability
* Support for both streaming and non-streaming (single) outputs

## Agents and Tools

In an [agentic](../../components/agents/index.md) workflow, a set of [tools](https://developer.nvidia.com/blog/introduction-to-llm-agents/#tools) are made available to the agent to use to perform a given task.

In NeMo Agent toolkit, both agents and tools are implemented as functions, because of this, an agent can be used as a tool for another agent allowing for multi-agent workflows (refer to the `examples/agents/mixture_of_agents` example for more details).

## Included Functions
For a complete list of functions run the following command:

```bash
nat info components -t function
```


## Key Concepts

### Type Safety

Functions use Python's type annotation system to:
- Validate inputs and outputs
- Convert between different types using converters
- Generate input and output schemas that provide runtime information about the function's input and output types

### Dual Output Modes

Functions support two output modes:
* **Single Output** - For operations that produce a single result
* **Streaming Output** - For operations that produce multiple results

A function can support either or both modes.

### Input and Output Schemas

Every function has schemas to define the input and output types. Every function has:
- An input schema
- A streaming output schema (optional)
- A single output schema (optional)

These schemas are Pydantic BaseModel classes that provide runtime validation and documentation. Pydantic models are used because they provide a way to validate and coerce values at runtime while also providing a way to document the schema properties of the input and output values.

### Asynchronous Operation

All function operations are asynchronous. To invoke a function, use one of the following methods:
- {py:meth}`~nat.builder.function.Function.ainvoke` - For single output operations
- {py:meth}`~nat.builder.function.Function.astream` - For streaming output operations

Using asynchronous operations allows for better performance and scalability when processing a large number of functions in parallel. In most cases, applications that integrate [LLMs](../llms/index.md) are IO bound and can benefit from cooperative multitasking. Asynchronous operations also provide a natural mechanism (using `ContextVar`s) for maintaining application state between multiple function invocations simultaneously.


## Writing Functions
For information about writing functions, refer to the [Writing Custom Functions](../../extend/custom-components/custom-functions/functions.md) document.

## Function Groups

Function groups are collections of related functions that can share configuration, context, and resources. For more information, refer to the [Function Groups](./function-groups.md) document.

