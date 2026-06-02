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

# NeMo Relay Telemetry Experiment

## Summary

This experiment wires NeMo Relay into the experimental Hermes Agent workflow for NVIDIA NeMo Agent Toolkit. The goal is to run an external CLI agent under NAT, capture the richer lifecycle telemetry that Relay sees around agent, LLM, and tool execution, and surface that telemetry through the normal NAT observability and evaluation paths.

The current prototype keeps NAT as the orchestration and exporter owner:

1. `nat run` or `nat eval` starts the `hermes_agent` workflow.
2. The workflow launches `nemo-relay run --agent hermes`.
3. Relay launches Hermes Agent and records ATOF lifecycle events to JSONL.
4. NAT reads the Relay ATOF file before the workflow returns.
5. NAT converts those Relay events into `IntermediateStep` objects.
6. Existing NAT exporters, including Phoenix, render the imported Relay spans as part of the NAT trace.

This is intentionally a bridge design rather than a Phoenix-only export path. It lets the same Relay telemetry feed NAT's trace exporters, profiler evaluators, ATIF output, and future non-Phoenix observability sinks.

## Harness Integration

### NAT Hermes Workflow Harness

The experimental Hermes adapter now has a `relay_enabled` mode. In direct mode, it still invokes Hermes single-query execution through the configured `command` and `command_args`. In Relay mode, the adapter builds a temporary Relay agent config, injects an observability plugin config for ATOF JSONL output, and invokes Relay instead.

Relevant workflow controls:

- `relay_enabled`: switches the adapter from direct Hermes execution to Relay-mediated execution.
- `relay_command`: points at `nemo-relay` or a local Relay checkout binary.
- `relay_atof_output_dir`: optionally preserves raw Relay ATOF output for debugging.

This keeps the workflow YAML small and makes local development easy with:

```bash
uv run nat run \
  --config_file examples/experimental/hermes_agent_adapter/configs/config-relay-phoenix.yml \
  --override workflow.relay_command ../NeMo-Flow/target/debug/nemo-relay \
  --input "Say hello in one sentence."
```

### NeMo Relay CLI And Hermes Hook Harness

Relay provides the process boundary and hook gateway. For Hermes specifically, the tricky part is that Hermes reads hooks from `.hermes/config.yaml` rather than receiving all hook behavior through command-line flags. Relay's transparent run path now temporarily merges hook-forwarding entries into the Hermes config, sets `HERMES_ACCEPT_HOOKS=1`, and restores the original file after the run.

Relay also needed Hermes-specific correlation fixes:

- Track `task_id` to route tool hooks back into the owning Hermes session.
- Avoid opening duplicate sessions for tool hooks that arrive with a task id rather than the main session id.
- Match post-tool events back to pre-tool events when Hermes changes or omits provider tool call ids.
- Preserve LLM request/response lifecycle events and token usage where available.

Those changes are what made Phoenix show the useful branch structure instead of only a top-level workflow span.

### Relay Python Exporter Harness

Relay now has a small Python-facing `NatTelemetryExporter`. It registers as a normal Relay subscriber and writes canonical ATOF event JSONL. The NAT side can replay that file into the active workflow trace.

This exporter is deliberately simple: it does not know about NAT internals, Phoenix, or evaluation. It only persists Relay events in a stable serialized form. That separation keeps Relay useful for other hosts while giving NAT a deterministic ingestion file.

### NAT Telemetry Bridge Harness

NAT gained an experimental bridge that maps Relay ATOF scope and mark events to `IntermediateStep` objects:

- Relay `agent` scopes become NAT task spans.
- Relay `llm` scopes become NAT LLM spans and carry usage metadata when present.
- Relay `tool` scopes become NAT tool spans.
- Relay mark events become zero-duration custom spans.
- Relay metadata is preserved under `nemo_relay` trace metadata for inspection.

The adapter injects the converted steps into the current NAT context before returning the final Hermes response. Because this happens inside the normal workflow lifecycle, existing NAT tracing exporters can consume the Relay spans without exporter-specific changes.

### Phoenix Visualization Harness

Phoenix worked well as the smoke-test UI because it makes missing hierarchy obvious. The first traces only showed outer workflow spans; after the Relay import and correlation fixes, Phoenix showed Hermes, LLM, and tool spans nested under the workflow. It also displayed LLM token totals from the imported Relay spans.

The Phoenix config lives in `configs/config-relay-phoenix.yml` and uses the standard NAT Phoenix exporter:

```yaml
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: nat-relay-hermes
```

### NAT Evaluation Harness

The eval path uses the same workflow wiring with a small JSON smoke dataset and runtime/profile evaluators. `nat eval` verified that imported Relay intermediate steps are available to NAT's evaluation output and profiler evaluators.

The eval config writes both workflow output and ATIF workflow output:

```bash
uv run nat eval \
  --config_file examples/experimental/hermes_agent_adapter/configs/config-relay-phoenix-eval.yml \
  --override workflow.relay_command ../NeMo-Flow/target/debug/nemo-relay
```

The smoke run completed successfully and produced:

- `workflow_output.json` with the generated answer and imported intermediate steps.
- `workflow_output_atif.json` with trajectory output.
- `avg_workflow_runtime_output.json`.
- `avg_num_llm_calls_output.json`.
- `avg_tokens_per_llm_end_output.json`.

Phoenix was not running during the eval verification, so the exporter logged connection-refused errors. The eval workflow still completed, which confirms the NAT eval pipeline, but a separate run with Phoenix listening is needed to verify UI upload for eval traces.

## What Went Well

- The existing NAT intermediate-step stream was a good integration point. Once Relay events are converted, Phoenix, ATIF output, and profiler evaluators can reuse NAT's normal telemetry path.
- Relay's ATOF lifecycle shape had enough structure to preserve agent, LLM, and tool hierarchy.
- The Relay plugin config path made it possible to enable ATOF output without asking users to permanently change their local Relay configuration.
- Phoenix gave fast visual feedback. It immediately exposed when we only had an outer workflow span, and later confirmed that nested Hermes, LLM, and tool spans were flowing.
- `nat run` and `nat eval` use the same adapter path, so the eval verification did not require a separate Relay integration.
- The local Relay override made branch development practical while preserving a future default of `relay_command: nemo-relay`.

## What Did Not Go Well

- The first user-facing error was just a missing `nemo-relay` command. The adapter now has a clearer error and the configs document the local binary override, but the setup path is still easy to miss.
- Initial Phoenix traces were too shallow. They showed NAT workflow spans but not the Relay/Hermes internals until the ATOF import and span parentage were fixed.
- Hermes hook installation is less direct than the other agent harnesses because hooks live in Hermes config files. Relay had to patch and restore that file for transparent runs.
- Hermes hook payloads do not always use the same identifiers across API and tool events. We had to add task-id session routing and tool name/argument fallback matching.
- One-shot Hermes execution can fail quietly when local Hermes model/provider setup is incomplete. The adapter needed more diagnostic text for empty stdout and launcher failures.
- Phoenix availability is external to NAT. If Phoenix is not running, NAT still completes, but export attempts produce connection errors and the UI trace is not available.

## Challenges And Difficulties

The largest challenge was boundary alignment. NAT, Relay, Hermes, and Phoenix each have a reasonable local model of execution, but they do not share the same span identifiers, session identifiers, or lifecycle vocabulary. The bridge had to preserve enough Relay metadata for debugging while still producing NAT-native `IntermediateStep` objects that existing exporters and evaluators understand.

The second challenge was process ownership. NAT owns the workflow invocation and exporter lifecycle, Relay owns the gateway and hook collection, and Hermes owns the actual CLI agent behavior. The prototype works by making each boundary explicit: NAT creates the workflow span, Relay writes ATOF, and NAT imports the file before closing the workflow.

The third challenge was deciding how much to specialize for Phoenix. Phoenix now supports direct ATIF upload, so a direct Relay-to-Phoenix path is possible. That would simplify Phoenix-only visualization, but it would bypass NAT's shared telemetry stream unless we also stitch trace context and evaluation artifacts back into NAT. For this experiment, the ATOF-to-IntermediateStep bridge is the more general path. A useful next step would be an optional direct ATIF/Phoenix export mode for comparison, while keeping the NAT bridge as the default path for `nat run`, `nat eval`, and non-Phoenix exporters.

## Current Status

The prototype has been verified with `nat run` and `nat eval`:

- `nat run` with Phoenix showed nested workflow, Hermes, LLM, and tool spans.
- `nat eval` completed with the Relay-enabled workflow and produced workflow output, ATIF output, and profiler evaluator results.
- Relay raw ATOF output contains paired start/end events for agent, LLM, and tool scopes.

The remaining work is mostly productization:

- Make the Relay command resolution and setup instructions smoother.
- Decide whether a Phoenix-direct ATIF upload mode should be offered as an optional fast path.
- Add broader examples beyond Hermes so the same pattern can be validated across the other experimental agent harnesses.
- Clarify the expected relationship between ATOF, ATIF, NAT intermediate steps, and provider-specific exporters in public documentation.
