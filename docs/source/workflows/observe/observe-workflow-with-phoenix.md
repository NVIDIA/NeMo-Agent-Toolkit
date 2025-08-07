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

# Observing a Workflow with Phoenix

This guide provides a step-by-step process to enable observability in a NeMo Agent toolkit workflow using Phoenix for tracing and logging. By the end of this guide, you will have:
- Configured telemetry in your workflow.
- Started the Phoenix server locally.
- Ability to view traces in the Phoenix UI.

### Step 1: Install the Phoenix Subpackage

Install the phoenix dependencies to enable tracing capabilities:

```bash
uv pip install -e '.[phoenix]'
```

### Step 2: Start the Phoenix Server

Run the following command to start Phoenix server locally:
```bash
phoenix serve
```
Phoenix should now be accessible at `http://0.0.0.0:6006`.

### Step 3: Modify Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:
```yaml
general:
  telemetry:
    tracing:
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: simple_calculator
```
This setup enables tracing through Phoenix at `http://localhost:6006/v1/traces`, with traces grouped into the `simple_calculator` project.

### Step 4: Run Your Workflow

From the root directory of the NeMo Agent toolkit library, install dependencies and run the pre-configured `simple_calculator_observability` example.

**Example:**
```bash
# Install the workflow and plugins
uv pip install -e examples/observability/simple_calculator_observability/

# Run the workflow with Phoenix telemetry settings
aiq run --config_file examples/observability/simple_calculator_observability/configs/config-phoenix.yml --input "What is 1*2?"
```
As the workflow runs, telemetry data will start showing up in Phoenix.

### Step 5: View Traces Data in Phoenix

- Open your browser and navigate to `http://0.0.0.0:6006`.
- Locate your workflow traces under your project name in projects.
- Inspect function execution details, latency, total tokens, request timelines and other info under Info and Attributes tab of an individual trace.

### Debugging

If you encounter issues while downloading the Phoenix package, try uninstalling and installing:
```bash
uv pip uninstall arize-phoenix

uv pip install arize-phoenix
```

After reinstalling, restart the Phoenix server:
```bash
phoenix serve
```

For more Arize-Phoenix details, view the documentation [here](https://docs.arize.com/phoenix).
