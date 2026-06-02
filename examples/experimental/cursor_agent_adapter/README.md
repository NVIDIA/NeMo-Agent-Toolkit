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

# Cursor Agent

This experimental NVIDIA NeMo Agent Toolkit example prototypes a primitive agent workflow type backed by Cursor Agent CLI.

The adapter is intentionally conservative:

- It accepts the same `ChatRequestOrMessage` input shape used by built-in workflow agents.
- Its config subclasses NAT's `AgentBaseConfig`; `llm_name` is optional because Cursor Agent manages model selection through `model`.
- Operational controls live in YAML config rather than an NAT `llms:` block.
- The example uses Cursor `plan` mode. It disables Cursor's CLI sandbox because sandboxing is not available on every local system; enable it in YAML when your Cursor Agent installation supports it.

## Installation And Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

Install and authenticate Cursor Agent CLI separately, then install this workflow:

```bash
cursor-agent login
cursor-agent status
uv pip install -e examples/experimental/cursor_agent_adapter
```

## Run The Workflow

From the repository root:

```bash
nat run \
  --config_file examples/experimental/cursor_agent_adapter/configs/config.yml \
  --input "Inspect the experimental Cursor agent adapter and summarize how it registers with NAT. Do not edit files."
```

## Direct Workflow

`configs/config.yml` wires the adapter directly as the workflow:

```yaml
workflow:
  _type: cursor_agent
  working_directory: .
  mode: plan
  sandbox: disabled
  trust_workspace: true
  timeout_seconds: 120
  max_output_chars: 12000
```

## Configurable Options

- `command`
- `working_directory`
- `llm_name` (accepted for NAT agent config consistency, unused by the CLI agent)
- `mode`
- `model`
- `sandbox`
- `trust_workspace`
- `timeout_seconds`
- `max_output_chars`
- `max_history`

## Authentication

The adapter does not require credentials in YAML. Cursor Agent CLI authentication is handled by the standalone `cursor-agent` command in the environment that launches `nat`.

Verify authentication from the same shell before running the workflow:

```bash
cursor-agent status
```

If `cursor-agent status` reports `Not logged in`, run:

```bash
cursor-agent login
```

For non-interactive environments, export `CURSOR_API_KEY` before starting `nat`. A Cursor desktop app login may not be visible to the standalone agent CLI.

## Workspace Trust

Cursor Agent requires workspace trust for headless `--print` runs. The example config sets `trust_workspace: true`, which passes `--trust` after you have reviewed and trusted this repository checkout. If you prefer not to trust through YAML, remove that field and run `cursor-agent` interactively once from the repository root to make the trust decision.

## Sandbox

Cursor Agent sandbox availability depends on the local installation and operating system. The example config sets `sandbox: disabled` so the headless smoke test works on systems where Cursor reports `Sandbox is unavailable`; `mode: plan` keeps the workflow in Cursor's read-only planning mode. If sandboxing works locally, set `sandbox: enabled`.

## Notes

The current agent implementation invokes `cursor-agent --print --output-format text` as a subprocess.
