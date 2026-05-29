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

# Claude Code Agent

This experimental NVIDIA NeMo Agent Toolkit example prototypes a primitive agent workflow type backed by the Claude Code Agent SDK.

The adapter is intentionally conservative:

- It accepts the same `ChatRequestOrMessage` input shape used by built-in workflow agents.
- Its config subclasses NAT's `AgentBaseConfig`; `llm_name` is optional because Claude Code Agent SDK manages model selection through `model`.
- Operational controls live in YAML config rather than an NAT `llms:` block.
- The default SDK permission mode is `plan`.

## Installation And Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

### Install This Workflow

From the root directory of the NeMo Agent Toolkit library, run:

```bash
uv pip install -e examples/experimental/code_agent_adapter
```

## Run The Workflow

For local end-to-end testing, authenticate Claude Code in the same shell or user profile that launches `nat`:

```bash
claude auth status
claude auth login
```

If `claude auth status` shows an active NVIDIA SSO-backed Claude login, run the live config:

```bash
nat run \
  --config_file examples/experimental/code_agent_adapter/configs/config.yml \
  --input "Inspect the experimental Claude Code agent adapter and summarize how it registers with NAT. Do not edit files."
```

`configs/config.yml` keeps `permission_mode: plan`, uses the `sonnet` model alias, caps `max_budget_usd`, and denies write-oriented tools for the first live smoke test.

## Direct Workflow

`configs/config.yml` wires the adapter directly as the workflow:

```yaml
workflow:
  _type: claude_code_agent
  permission_mode: plan
  model: sonnet
  working_directory: .
  setting_sources: [project]
  max_turns: 10
  max_budget_usd: 1.00
  disallowed_tools:
    - Bash
    - Edit
    - MultiEdit
    - NotebookEdit
    - Write
```

## Configurable Options

- `working_directory`
- `llm_name` (accepted for NAT agent config consistency, unused by the SDK agent)
- `permission_mode`
- `model`
- `append_system_prompt`
- `allowed_tools`
- `disallowed_tools`
- `setting_sources`
- `additional_directories`
- `max_turns`
- `max_budget_usd`
- `timeout_seconds`
- `max_output_chars`
- `max_history`

## Authentication

The adapter does not require credentials in YAML. In live mode, the Claude Agent SDK follows Claude Code authentication in the launching environment. For local testing, an NVIDIA SSO-backed Claude Code login can be used if `claude auth status` reports an active login in the same shell/user profile. Stale `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` environment variables can take precedence over that login.

## Notes

Live execution should stay opt-in until permission, sandboxing, credentials, and cost controls are reviewed. The current agent implementation uses `claude_agent_sdk.query()` and `ClaudeAgentOptions`.
