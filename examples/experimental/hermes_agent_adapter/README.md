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

# Hermes Agent

This experimental NVIDIA NeMo Agent Toolkit example prototypes a primitive agent workflow type backed by Hermes Agent CLI.

The adapter is intentionally conservative:

- It accepts the same `ChatRequestOrMessage` input shape used by built-in workflow agents.
- Its config subclasses NAT's `AgentBaseConfig`; `llm_name` is optional because Hermes manages provider/model selection through local configuration, `provider`, and `model`.
- Operational controls live in YAML config rather than an NAT `llms:` block.
- The adapter uses Hermes one-shot mode so stdout contains the final response text.

## Installation And Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/get-started/installation.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.

Install this workflow package:

```bash
uv pip install -e examples/experimental/hermes_agent_adapter
```

The default config launches Hermes Agent with `uvx`, so a global `hermes` executable is not required. You still need to configure Hermes credentials before a live model-backed run succeeds:

```bash
uvx --from hermes-agent hermes setup
uvx --from hermes-agent hermes model
```

If you prefer to install Hermes Agent once and use the `hermes` command directly, either use `pipx`:

```bash
pipx install hermes-agent
hermes setup
hermes --version
```

or use the upstream installer:

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

After installing `hermes` directly, open a fresh shell or reload your shell profile, then verify `hermes --version` from the same shell that will run `nat`.

## Run The Workflow

From the repository root:

```bash
nat run \
  --config_file examples/experimental/hermes_agent_adapter/configs/config.yml \
  --input "Inspect the experimental Hermes agent adapter and summarize how it registers with NAT. Do not edit files."
```

The default config resolves Hermes Agent from PyPI through uv, so the first run may download the package.

If you installed `hermes` directly on `PATH`, use the installed-CLI config instead:

```bash
nat run \
  --config_file examples/experimental/hermes_agent_adapter/configs/config-installed.yml \
  --input "Inspect the experimental Hermes agent adapter and summarize how it registers with NAT. Do not edit files."
```

## Direct Workflow

`configs/config.yml` wires the adapter directly as the workflow through `uvx`:

```yaml
workflow:
  _type: hermes_agent
  command: uvx
  command_args: ["--from", "hermes-agent", "hermes"]
  working_directory: .
  timeout_seconds: 300
  max_output_chars: 12000
```

`configs/config-installed.yml` uses an installed `hermes` command:

```yaml
workflow:
  _type: hermes_agent
  command: hermes
  working_directory: .
  timeout_seconds: 300
  max_output_chars: 12000
```

## Configurable Options

- `command`
- `command_args`
- `working_directory`
- `llm_name` (accepted for NAT agent config consistency, unused by the CLI agent)
- `provider`
- `model`
- `timeout_seconds`
- `max_output_chars`
- `max_history`
- `error_on_empty_output`

## Authentication

The adapter does not require credentials in YAML. Hermes authentication and provider configuration are handled by the local Hermes installation in the environment that launches `nat`.

## Notes

The current agent implementation invokes Hermes one-shot mode (`hermes -z`) through the configured command. For launcher-based installs, use `command_args`. For example:

```yaml
workflow:
  _type: hermes_agent
  command: uvx
  command_args: ["--from", "hermes-agent", "hermes"]
```

Hermes one-shot mode suppresses intermediate logs and writes only the final agent response to stdout. If the workflow reports that Hermes exited successfully but printed no final response, verify Hermes outside NAT:

```bash
uvx --from hermes-agent hermes status
uvx --from hermes-agent hermes setup
uvx --from hermes-agent hermes model
uvx --from hermes-agent hermes -z "Say hello in one sentence."
```

`hermes status` should show a concrete model and an authenticated provider. If it reports `Model: (not set)` or missing credentials, finish Hermes provider setup first. When using a non-default provider, set both `provider` and `model` in the workflow config. Hermes treats `--provider` without `--model` as ambiguous in one-shot mode.
