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
- The adapter uses Hermes single-query mode so stdout contains the final response text.

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
uvx --from hermes-agent hermes status
uvx --from hermes-agent hermes chat -q "Say hello in one sentence." -Q
```

`hermes status` should show a concrete model and authenticated provider, and the `chat -q` command should print either a response or a provider/authentication error. If it reports an error such as `model does not exist`, select a model that your configured provider endpoint supports before running `nat`.

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

If the workflow reports that Hermes exited successfully but printed no final response text, run the Hermes preflight commands from [Installation And Setup](#installation-and-setup). That error means Hermes launched, but the local Hermes provider/model/auth configuration is not ready to return a model response.

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
- `relay_enabled`
- `relay_command`
- `relay_atof_output_dir`

## NeMo Relay Telemetry

Set `relay_enabled: true` to launch Hermes through NeMo Relay and import Relay's ATOF lifecycle events into the active NeMo Agent Toolkit telemetry stream. The adapter creates a temporary Relay plugin config with the ATOF exporter enabled, runs `nemo-relay run --agent hermes`, then maps the emitted Relay scope and mark events into NAT intermediate steps before the workflow returns.

```yaml
workflow:
  _type: hermes_agent
  command: uvx
  command_args: ["--from", "hermes-agent", "hermes"]
  relay_enabled: true
  relay_atof_output_dir: ./.tmp/nat-relay-hermes-atof
```

Install or expose the NeMo Relay CLI as `nemo-relay`, or set `relay_command` to an absolute path. For local Relay branch development from a sibling checkout, build the CLI from the Relay repository:

```bash
cd ../NeMo-Flow
cargo build -p nemo-relay-cli --bin nemo-relay
cd ../nemo-agent-toolkit
```

Then run the Relay-enabled workflow:

```bash
nat run \
  --config_file examples/experimental/hermes_agent_adapter/configs/config-relay.yml \
  --override workflow.relay_command ../NeMo-Flow/target/debug/nemo-relay \
  --override workflow.relay_atof_output_dir ./.tmp/nat-relay-hermes-atof \
  --input "Say hello in one sentence."
```

Set `relay_atof_output_dir` while debugging to keep Relay's raw `events.jsonl` after the workflow finishes:

```bash
cat ./.tmp/nat-relay-hermes-atof/events.jsonl
```

NeMo Agent Toolkit telemetry exporters under `general.telemetry.tracing` receive both the outer toolkit workflow spans and the imported Relay agent/tool/LLM spans.

## Phoenix With NeMo Relay

Install the Phoenix integration and start Phoenix:

```bash
uv pip install -e packages/nvidia_nat_phoenix
docker run -it --rm -p 4317:4317 -p 6006:6006 arizephoenix/phoenix:13.22
```

In another terminal, run the Relay/Phoenix config:

```bash
nat run \
  --config_file examples/experimental/hermes_agent_adapter/configs/config-relay-phoenix.yml \
  --override workflow.relay_command ../NeMo-Flow/target/debug/nemo-relay \
  --input "Inspect examples/experimental/hermes_agent_adapter/README.md and summarize the NeMo Relay Telemetry section."
```

Open `http://localhost:6006` and select the `nat-relay-hermes` project. The trace should include the NAT workflow span plus imported Relay/Hermes agent, LLM, and tool spans.

The evaluation smoke config uses the same Relay bridge and writes ATIF output:

```bash
nat eval \
  --config_file examples/experimental/hermes_agent_adapter/configs/config-relay-phoenix-eval.yml \
  --override workflow.relay_command ../NeMo-Flow/target/debug/nemo-relay
```

Eval outputs are written under `./.tmp/nat/examples/hermes_agent_adapter/relay_phoenix_eval/`.

See [RELAY_TELEMETRY_EXPERIMENT.md](RELAY_TELEMETRY_EXPERIMENT.md) for the implementation notes and validation summary.

## Authentication

The adapter does not require credentials in YAML. Hermes authentication and provider configuration are handled by the local Hermes installation in the environment that launches `nat`.

## Notes

The current agent implementation invokes Hermes single-query mode (`hermes chat -q ... -Q`) through the configured command. For launcher-based installs, use `command_args`. For example:

```yaml
workflow:
  _type: hermes_agent
  command: uvx
  command_args: ["--from", "hermes-agent", "hermes"]
```

Hermes single-query mode prints useful provider errors when local model/provider setup is wrong. If the workflow reports that Hermes exited successfully but printed no final response, verify Hermes outside NAT:

```bash
uvx --from hermes-agent hermes status
uvx --from hermes-agent hermes setup
uvx --from hermes-agent hermes model
uvx --from hermes-agent hermes chat -q "Say hello in one sentence." -Q
```

`hermes status` should show a concrete model and an authenticated provider. If it reports `Model: (not set)`, missing credentials, or the `chat -q` command reports `model does not exist`, finish Hermes provider/model setup first. When using a non-default provider, set both `provider` and `model` in the workflow config.
