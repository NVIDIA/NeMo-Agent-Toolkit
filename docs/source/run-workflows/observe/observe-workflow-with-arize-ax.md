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

# NVIDIA NeMo Agent Toolkit: Observing a Workflow with Arize AX

This guide shows how to send **OpenTelemetry** traces from NeMo Agent Toolkit to [Arize AX](https://arize.com/docs/ax/) using the built-in `arize_ax` exporter (`nvidia-nat[opentelemetry]`). For field reference and custom OTLP endpoints, see [Adding Telemetry Exporters — Arize AX](../../extend/custom-components/telemetry-exporters.md).

## Step 1: Arize space and API key

In the Arize AX UI, open your **space** and create or select a **project** for trace ingestion. You need the **space ID** and an **API key** with permission to send OTLP data. Official OTLP details: [OpenTelemetry with Arize OTel](https://arize.com/docs/ax/integrations/opentelemetry/opentelemetry-arize-otel).

## Step 2: Configure the environment

```bash
export ARIZE_SPACE_ID="<your-space-id>"
export ARIZE_API_KEY="<your-api-key>"
# Optional: overrides the default in config-arize-ax.yml (${ARIZE_PROJECT_NAME:-simple_calculator})
export ARIZE_PROJECT_NAME="simple_calculator"
```

For **EU** data residency, set `use_eu_region: true` under `general.telemetry.tracing.arize_ax` in your config (see the example file below).

## Step 3: Install the OpenTelemetry extra

```bash
uv pip install -e ".[opentelemetry]"
# or, from PyPI: uv pip install "nvidia-nat[opentelemetry]"
```

## Step 4: Run the simple calculator observability example

From the root of the NeMo Agent Toolkit repository:

```bash
uv pip install -e examples/observability/simple_calculator_observability/

nat run --config_file examples/observability/simple_calculator_observability/configs/config-arize-ax.yml --input "What is 2 * 4?"
```

You should see a log line such as `Started exporter 'arize_ax'`. Open the Arize project matching `ARIZE_PROJECT_NAME` to view traces.

## Related configuration

- Example config: `examples/observability/simple_calculator_observability/configs/config-arize-ax.yml`
- [Telemetry exporters reference (Arize AX)](../../extend/custom-components/telemetry-exporters.md)
