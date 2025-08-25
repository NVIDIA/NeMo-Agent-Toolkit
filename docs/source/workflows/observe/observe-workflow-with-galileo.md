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

This guide provides a step-by-step process to enable observability in a NeMo Agent toolkit workflow using Galileo for tracing. By the end of this guide, you will have:

- Configured telemetry in your workflow.
- Ability to view traces in the Galileo platform.

## Step 1: Sign up for Galileo

- Visit [https://app.galileo.ai/](https://app.galileo.ai/) to create your account or sign in.

## Step 2: Create a Project and Log Stream

After logging in:

- Create a new **Logging** project (or reuse an existing one).
- Inside the project create (or locate) the **Log Stream** you will write to.

## Step 3: Generate API Key

Go to **Settings → API Keys** to generate a new API key and copy it.

You will need the following values:

- `Galileo-API-Key`
- `project` (project name)
- `logstream` (log-stream name)


### Step 4: Configure Your Environment
Set the following environment variables in your terminal
```bash
export GALILEO_API_KEY=<your_api_key>
```

## Step 5: Install the OpenTelemetry Subpackage

```bash
uv pip install '.[opentelemetry]'
```

## Step 6: Modify Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:

```yaml
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
        api_key: ${GALILEO_API_KEY}
```

## Step 7: Run Your Workflow

From the root directory of the NeMo Agent toolkit library, install dependencies and run the pre-configured `simple_calculator_observability` example.

**Example:**

```bash
# Install the workflow and plugins
uv pip install -e examples/observability/simple_calculator_observability/

# Run the workflow with Galileo telemetry settings
# Note, you may have to update configuration settings based on your Galileo account
nat run --config_file examples/observability/simple_calculator_observability/configs/config-galileo.yml --input "What is 1*2?"
```

As the workflow runs, telemetry data will start showing up in Galileo.

## Step 8: View Traces Data in Galileo

- Open your browser and navigate to [https://app.galileo.ai/](https://app.galileo.ai/).
- Select your project and navigate to **View all logs**.
- Inspect function execution details, latency, total tokens, request timelines and other info within individual traces.
- New traces should appear within a few seconds.



For additional help, see the [Galileo OpenTelemetry integration docs](https://v2docs.galileo.ai/integrations/otel).
