---
name: nat-tools-and-functions
description: Use when authoring, registering, composing, or testing custom NeMo Agent toolkit tools, functions, function groups, Python components, custom agents, custom evaluators, or advanced extension patterns.
metadata:
  version: "0.1.0"
  status: initial
---

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

# NeMo Agent toolkit Tools and Functions

Use this skill when adding custom Python behavior to the toolkit.

## Workflow

1. Read `references/tools-and-functions.md` for the registration pattern.
2. Use `FunctionInfo.from_fn()` for simple async functions.
3. Use function groups when related tools share a resource.
4. Keep heavyweight optional imports lazy.
5. Add focused tests for new component behavior.

## References

- `references/tools-and-functions.md`
- `references/advanced-python.md`
