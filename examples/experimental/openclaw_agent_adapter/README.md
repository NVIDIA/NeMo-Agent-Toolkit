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

# OpenClaw Agent

This experimental NVIDIA NeMo Agent Toolkit example prototypes a primitive agent workflow type backed by the OpenClaw CLI.

The adapter is intentionally conservative:

- It accepts the same `ChatRequestOrMessage` input shape used by built-in workflow agents.
- Its config subclasses NAT's `AgentBaseConfig`; `llm_name` is optional because OpenClaw manages model selection through local config and `model`.
- Operational controls live in YAML config rather than an NAT `llms:` block.
- The workflow calls `openclaw agent --json --local` by default for a local one-shot run. Set `local: false` to route through a running OpenClaw Gateway.

## Installation And Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

Install this workflow package:

```bash
uv pip install -e examples/experimental/openclaw_agent_adapter
```

Install and configure the OpenClaw CLI separately:

```bash
npm install -g openclaw@latest
openclaw onboard
```

## Run The Workflow

Run from the repository root:

```bash
nat run \
  --config_file examples/experimental/openclaw_agent_adapter/configs/config.yml \
  --input "Inspect the experimental OpenClaw agent adapter and summarize how it registers with NAT. Do not edit files."
```

## Direct Workflow

`configs/config.yml` wires the adapter directly as the workflow:

```yaml
workflow:
  _type: openclaw_agent
  working_directory: .
  agent_id: main
  session_key: nat-openclaw
  local: true
  codex_app_server_mode: guardian
  codex_app_server_approval_policy: on-request
  codex_app_server_sandbox: workspace-write
  agent_timeout_seconds: 600
  timeout_seconds: 660
  max_output_chars: 12000
```

## Configurable Options

- `command`
- `working_directory`
- `llm_name` (accepted for NAT agent config consistency, unused by the OpenClaw CLI)
- `agent_id`
- `session_key`
- `model`
- `thinking`
- `local`
- `codex_app_server_mode`
- `codex_app_server_approval_policy`
- `codex_app_server_sandbox`
- `agent_timeout_seconds`
- `timeout_seconds`
- `max_output_chars`
- `max_history`

## Authentication

The adapter does not require credentials in YAML. Authentication and provider selection are inherited from the OpenClaw installation used by the `openclaw` command.

OpenClaw local runs use the current user's OpenClaw state under `~/.openclaw`. Run `nat` from the same shell/user profile where `openclaw doctor` and `openclaw agent --local` work. Restricted sandboxes that cannot read or write that state can surface misleading Codex harness errors such as `timed out waiting for cloud requirements`.

## Codex Harness Policy

When OpenClaw uses its Codex harness, the Codex cloud requirements can reject full-access runs that map to `approval_policy: Never`. The example therefore defaults the OpenClaw subprocess environment to:

```text
OPENCLAW_CODEX_APP_SERVER_MODE=guardian
OPENCLAW_CODEX_APP_SERVER_APPROVAL_POLICY=on-request
OPENCLAW_CODEX_APP_SERVER_SANDBOX=workspace-write
```

If you previously ran this example with `session_key: nat`, use the updated `session_key: nat-openclaw` or choose another fresh session key so OpenClaw does not reuse a persisted Codex thread created with the old policy.

## Notes

OpenClaw documents an App SDK import path, `@openclaw/sdk`, but that package is not currently published to the public npm registry. The upstream source package is marked private, so this example uses the stable public CLI surface instead of depending on an install path that fails with npm 404.

The OpenClaw CLI owns Gateway connection behavior, workspace/runtime selection, provider credentials, and most approval behavior. This adapter only sets the Codex app-server environment overrides above because they are necessary for non-interactive NAT runs against cloud-constrained Codex accounts.
