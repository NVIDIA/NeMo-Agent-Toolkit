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
- Run the Phoenix server.
- Able to view traces in the Phoenix UI.


### Step 1: Modify Workflow Configuration

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
      phoenix:
        _type: phoenix
        endpoint: http://localhost:6006/v1/traces
        project: simple_calculator
```
This setup enables:
- Console logging with WARN level messages.
- Tracing through Phoenix at `http://localhost:6006/v1/traces`.

### Step 2: Start the Phoenix Server
Run the following command to start Phoenix server locally:
```bash
phoenix serve
```
Phoenix should now be accessible at `http://0.0.0.0:6006`.

### Step 3: Run Your Workflow
From the root directory of the NeMo Agent toolkit library, install dependencies and execute your workflow.

**Example:**
```bash
uv pip install -e examples/basic/functions/simple_calculator
```
As the workflow runs, telemetry data will start showing up in Phoenix and the console.

### Step 4: View Traces Data in Phoenix
- Open your browser and navigate to `http://0.0.0.0:6006`.
- Locate your workflow traces under the **Traces** section in projects.
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

For more Arize-Phoenix details view doc [here](https://docs.arize.com/phoenix)
