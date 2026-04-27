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

# Demo workflow

**Complexity:** 🟢 Beginner

Minimal NeMo Agent Toolkit project named `demo_workflow`. The **ReAct (Reasoning and Acting) agent** (`workflow._type: react_agent`, implemented with LangChain and LangGraph via `nvidia-nat-langchain`) uses:

- **Core time tools** (`nvidia-nat-core`): `current_datetime`, `current_timezone`
- **Wikipedia** (`nvidia-nat-langchain` `wiki_search`): short article snippets for places, people, or topics (needs outbound network to Wikipedia)

`config.yml` includes **Arize AX** OTLP tracing (`_type: arize_ax`) so each `nat run` can send traces to your Arize project when credentials are set (see below). There is no custom Python function in this package; only YAML wiring.

## Install

From the NeMo Agent Toolkit repository root:

```bash
uv pip install -e examples/demo_workflow
```

## API keys and environment

**NIM (required for the default model):** see the [quick start](https://docs.nvidia.com/nemo/agent-toolkit/latest/get-started/index.html).

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

**Arize AX (required for traces to appear in Arize):** use the space ID, API key, and project name from the Arize AX UI. See [OpenTelemetry (arize-otel)](https://arize.com/docs/ax/integrations/opentelemetry/opentelemetry-arize-otel).

```bash
export ARIZE_SPACE_ID="<your-space-id>"
export ARIZE_API_KEY="<your-api-key>"
export ARIZE_PROJECT_NAME="demo_workflow"
```

`ARIZE_PROJECT_NAME` defaults to `demo_workflow` in `config.yml` if unset (`${ARIZE_PROJECT_NAME:-demo_workflow}`). For EU data residency, set `use_eu_region: true` under `general.telemetry.tracing.arize_ax` in `config.yml`.

## Run

```bash
nat run --config_file examples/demo_workflow/src/nat_demo_workflow/configs/config.yml --input "What is the current date and time in my timezone?"
```

**Wikipedia** (needs network):

```bash
nat run --config_file examples/demo_workflow/src/nat_demo_workflow/configs/config.yml --input "What is Paris France known for, in one sentence?"
```

You should see `Started exporter 'arize_ax'` in the log when Arize credentials are present. Open the Arize project matching `ARIZE_PROJECT_NAME` (or `demo_workflow`) to view traces.

## Example queries (tool-calling coverage)

Replace the `--input` string. These are meant to exercise **different tools** or **multiple tools** in one run; check logs or Arize for `wikipedia_search`, `current_timezone`, and `current_datetime`.

1. **Wikipedia only** — `In one sentence, who was Alan Turing?`
2. **Timezone only** — `What IANA timezone name is this run using? Reply with only the zone name.`
3. **Time after zone** — `What is the current local date and time for this session? Cite the timezone you used.`
4. **Wikipedia, structured** — `Use Wikipedia: in two sentences, what is the Large Hadron Collider?`
5. **Multi-tool** — `What is the current UTC time, and in one sentence who was Marie Curie?`

## Validate

```bash
nat validate --config_file examples/demo_workflow/src/nat_demo_workflow/configs/config.yml
```
