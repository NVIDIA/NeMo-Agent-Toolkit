<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Observing a Workflow with Galileo

This guide shows how to stream OpenTelemetry (OTel) traces from your AIQ toolkit workflows to **Galileo** so that you can analyse performance, token usage, and latency directly in the Galileo console.

In this guide, you will learn how to:

- Create a project and log-stream in Galileo and generate an API key.
- Configure your workflow (YAML) or Python script to export traces to Galileo.
- Run the workflow and view traces in the Galileo UI.

---

### Step 1. Create a project & API key in Galileo

1. Sign in to the [Galileo console](https://app.galileo.ai/).
2. Create a new **Logging** project (or reuse an existing one).
3. Inside the project create (or locate) the **Log Stream** you will write to.
4. From **Settings → API Keys** generate a new API key and copy it.

You will need the following values in the next steps:

- `Galileo-API-Key`
- `project` (project name)
- `logstream` (log-stream name)

### Step 2: Modify Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:
```bash
general:
  telemetry:
    logging:
      console:
        _type: console
        level: WARN
    tracing:
      galileo:
        _type: galileo
        # Cloud endpoint – change if you are using an on-prem cluster.
        endpoint: https://app.galileo.ai/api/galileo/otel/traces
        project: simple_calculator
        logstream: default
        api_key: "<YOUR-GALILEO-API-KEY>"
```

### Step 3. Run the workflow

```bash
uv pip install -e examples/simple_calculator
aiq run --config_file examples/simple_calculator/configs/config.yml --input "2 + 2"
```

As the workflow runs, spans are exported to Galileo and (optionally) printed to your console.
Open the [Galileo console](https://app.galileo.ai/), select your project, and navigate to **Traces**. New traces should appear within a few seconds.

### Debugging
Galileo relies on the OpenInference libraries. Which are included in the following dependency:

```bash
uv pip install arize-phoenix
```

For additional help see the [Galileo OpenTelemetry integration docs](https://v2docs.galileo.ai/integrations/otel).
