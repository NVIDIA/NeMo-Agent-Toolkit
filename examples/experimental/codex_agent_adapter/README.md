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

# Codex Agent

This experimental NVIDIA NeMo Agent Toolkit example prototypes a primitive agent workflow type backed by the Codex SDK.

The adapter is intentionally conservative:

- It accepts the same `ChatRequestOrMessage` input shape used by built-in workflow agents.
- Its config subclasses NAT's `AgentBaseConfig`; `llm_name` is optional because Codex manages model selection through `model`.
- Operational controls live in YAML config rather than an NAT `llms:` block.
- The default Codex sandbox mode is `read-only`, with `approval_policy: never` for non-interactive `nat run` execution.

## Installation And Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

Install this workflow and the Codex SDK package:

```bash
uv pip install -e examples/experimental/codex_agent_adapter
npm install --prefix examples/experimental/codex_agent_adapter
```

The SDK runner uses Node.js 18 or later. If the Node dependencies are installed somewhere else, set `node_package_directory` in the workflow config to the directory containing `node_modules`.

## Run The Workflow

From the repository root:

```bash
nat run \
  --config_file examples/experimental/codex_agent_adapter/configs/config.yml \
  --input "Inspect the experimental Codex agent adapter and summarize how it registers with NAT. Do not edit files."
```

## Direct Workflow

`configs/config.yml` wires the adapter directly as the workflow:

```yaml
workflow:
  _type: codex_agent
  working_directory: .
  sandbox_mode: read-only
  approval_policy: never
  skip_git_repo_check: false
  timeout_seconds: 300
  max_output_chars: 12000
```

## Configurable Options

- `node_command`
- `node_package_directory`
- `working_directory`
- `llm_name` (accepted for NAT agent config consistency, unused by the SDK agent)
- `codex_path_override`
- `base_url`
- `codex_config`
- `thread_id`
- `model`
- `sandbox_mode`
- `skip_git_repo_check`
- `approval_policy`
- `model_reasoning_effort`
- `network_access_enabled`
- `web_search_mode`
- `web_search_enabled`
- `additional_directories`
- `timeout_seconds`
- `max_output_chars`
- `max_history`

## Authentication

The adapter does not require credentials in YAML. The Codex SDK follows Codex authentication in the environment that launches `nat`, and it can also inherit standard Codex/OpenAI environment variables from that process.

For local diagnostics, run:

```bash
examples/experimental/codex_agent_adapter/node_modules/.bin/codex doctor
```

If `nat run` times out, the adapter reports recent Codex SDK stream events in the error. A timeout with no completed agent message usually means Codex is still working, waiting on an approval path, or blocked by local Codex connectivity/authentication.

## Notes

The current agent implementation uses the TypeScript `@openai/codex-sdk` package from a small Node.js runner. The Python workflow remains the NAT component boundary and translates NAT input/output into SDK calls.
