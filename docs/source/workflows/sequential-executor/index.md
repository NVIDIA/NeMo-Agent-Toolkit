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

# About the Sequential Executor
A sequential executor is a control flow component that chains multiple functions together, where each function's output becomes the input for the next function. You can opt to validate the compatibility of the output of one function and the input type of the next function in the chain. This creates a linear tool execution pipeline that executes functions in a predetermined sequence without requiring LLMs or agents for orchestration. The sequential executor process allows for better error handling. 

Additionally, you can customize prompts, such as streaming support and compatibility validation, in your YAML config files for your specific needs. 

```{toctree}
:hidden:
:caption: Sequential Executor
Configure Sequential Executor<./sequential-executor.md>
```