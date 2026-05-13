---
name: nat-mcp-and-serving
description: Use when serving NeMo Agent toolkit workflows, exposing workflows through FastAPI, configuring MCP clients or servers, or troubleshooting transport and server setup.
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

# NeMo Agent toolkit MCP and Serving

Use this skill when a workflow needs to call remote MCP tools or be served as an API.

## Workflow

1. Read the reference for the protocol or server target.
2. Keep serving configuration separate from workflow logic where possible.
3. Validate locally with a small request before adding deployment details.
4. Prefer documented `nat serve` and `nat start` commands over ad hoc servers.

## References

- `references/mcp.md`
- `references/fastapi-frontend.md`
