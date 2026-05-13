---
name: nat-installation
description: Use when installing or configuring NVIDIA NeMo Agent toolkit, verifying the `nat` CLI, setting up optional extras, or creating a first hello-world workflow.
metadata:
  version: "0.1.1"
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

# NeMo Agent toolkit Installation

Use this skill when the task is about setup, dependencies, package extras, environment variables, or a first runnable workflow.

## Workflow

1. Read `references/installation.md`.
2. Prefer the smallest package extra that supports the requested workflow.
3. Verify the CLI with `uv run nat --version` or `uv run nat --help`.
4. For a first workflow, adapt `references/hello_world.yaml`.

## Key Commands

```bash
uv run nat --help
uv run nat --version
uv run nat info components
```

## References

- `references/installation.md`
- `references/hello_world.yaml`
