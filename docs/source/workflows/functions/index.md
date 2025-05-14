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

# Functions

Functions (tools) are the main building blocks of AIQ Toolkit and define the logic of your workflow.

In AIQ Toolkit, functions are a core abstraction that offer type-safe, asynchronous operations with support for both single and streaming outputs. They wrap callable objects (like Python functions or coroutines) and enhance them with:

* Type validation and conversion
* Schema-based input/output validation via Pydantic models
* Unified interfaces to improve composability
* Support for both streaming and non-streaming (single) outputs

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
- {py:meth}`~aiq.builder.function.Function.ainvoke` - For single output operations
- {py:meth}`~aiq.builder.function.Function.astream` - For streaming output operations

Using asynchronous operations allows for better performance and scalability when processing a large number of functions in parallel. In most cases, applications that integrate LLMs are IO bound and can benefit from cooperative multitasking. Asynchronous operations also provide a natural mechanism (using `ContextVar`s) for maintaining application state between multiple function invocations simultaneously.


## Writing Functions
For information about writing functions, refer to the [Writing Custom Functions](../../extend/functions.md) document.


## Using the Code Execution Function
```{toctree}
./code-execution.md
```
