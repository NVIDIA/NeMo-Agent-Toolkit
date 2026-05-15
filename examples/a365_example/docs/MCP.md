<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 Tooling

This example no longer includes a custom MCP server.

The original draft example carried a self-hosted `graph_mail` MCP server to
demonstrate tool wiring. That custom server was intentionally pruned from the
committed example so the repo does not imply that Microsoft Agent 365
deployments require bespoke MCP implementations for Microsoft workloads.

## Current Direction

For the A365 / tenant-managed tooling path, prefer Microsoft’s current Work IQ
tooling documentation:

- Work IQ MCP overview:
  <https://learn.microsoft.com/en-us/microsoft-agent-365/tooling-servers-overview>
- Work IQ / tooling setup guidance:
  <https://learn.microsoft.com/en-us/microsoft-agent-365/developer/tooling>
- A365 CLI setup reference:
  <https://learn.microsoft.com/en-us/microsoft-agent-365/developer/reference/cli/setup>

## What This Example Covers

This repo example is intentionally narrower. It covers:

- Teams / Azure Bot front-end wiring
- worker deployment shape
- telemetry configuration
- setup and troubleshooting surfaces

It does not attempt to demonstrate the full Work IQ / managed-tooling path in
this PR.

## If You Need Managed Tooling

Treat managed tooling / Work IQ support as a follow-up track:

- verify the services available in your tenant
- verify the required permissions and licensing
- verify the NeMo Agent Toolkit configuration and documentation needed to consume those
  services cleanly
