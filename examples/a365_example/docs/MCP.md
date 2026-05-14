<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 MCP

This example keeps MCP scope intentionally narrow.

## Graph Mail MCP Server Used By The Demo

The strongest end-to-end demo path uses a local or separately deployed Graph
mail MCP server owned by this example:

- `graph_mail`

Those are wired directly into the main deployment config:

- [../configs/a365_worker.yml](../configs/a365_worker.yml)

This is the path the Teams-triggered worker demo uses today.

In a real deployment, point the worker config at your own Graph mail MCP
endpoint, for example:

- `https://<graph-mail-mcp-host>/mcp`

## What To Validate

On startup, look for log lines such as:

- `Registered MCP server`
- `graph_mail`

For the local MCP path, also confirm the configured function groups and the
workflow tool names line up with the available server endpoints.

Keep this example intentionally narrow. If you want to add more MCP tools,
start from the same `mcp_client` pattern used by `graph_mail` and add them only
after the primary worker + Teams + telemetry + mail-tool path is working.
