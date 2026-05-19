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
