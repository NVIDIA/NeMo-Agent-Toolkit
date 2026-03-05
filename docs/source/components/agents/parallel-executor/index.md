<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# About the Parallel Executor
A parallel executor is a deterministic control flow component that executes multiple tools concurrently with a shared input and returns appended branch outputs as text blocks. Use it when branch tools are independent and can run in parallel.

Like the sequential executor, the parallel executor can be configured either as a workflow or as a function.

```{toctree}
:hidden:
:caption: Parallel Executor

Configure Parallel Executor<./parallel-executor.md>
```
