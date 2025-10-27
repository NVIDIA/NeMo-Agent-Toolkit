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

# Observing a Workflow with DBNL

This guide provides a step-by-step process to enable observability in a NeMo Agent toolkit workflow using DBNL for tracing. By the end of this guide, you will have:

- Configured telemetry in your workflow.
- Ability to view traces in the DBNL platform.

## Step 1: Install DBNL

Visit [https://docs.dbnl.com/get-started/quickstart](https://docs.dbnl.com/get-started/quickstart) to install DBNL.

## Step 2: Create a Project

Create a new Trace Ingestion project in DBNL. To create a new project in DBNL:

1. Navigate to your DBNL deployment (e.g. <http://localhost:8080/>)
2. Go to Projects > + New Project
3. Name your project `nat-calculator`
4. Add a LLM connection to your project
5. Select Trace Ingestion as the project Data Source
6. Click on Generate API Token and note down the generated **API Token**
7. Note down the **Project Id** for the project

## Step 3: Configure Your Environment

Set the following environment variables in your terminal:

```bash
# DBNL_API_URL should point to your deployment API URL (e.g. http://localhost:8080/api)
export DBNL_API_URL=<your_api_url>
export DBNL_API_TOKEN=<your_api_token>
export DBNL_PROJECT_ID=<your_project_id>
```

## Step 4: Install the NeMo Agent toolkit OpenTelemetry Subpackages

```bash
# Install specific telemetry extras required for DBNL
uv pip install -e '.[opentelemetry]'
```

## Step 5: Modify NeMo Agent toolkit Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:
```yaml
general:
  telemetry:
    tracing:
      dbnl:
        _type: dbnl

## Step 6: Run the workflow

From the root directory of the NeMo Agent toolkit library, install dependencies and run the pre-configured `simple_calculator_observability` example.

**Example:**

```bash
# Install the workflow and plugins
uv pip install -e examples/observability/simple_calculator_observability/

# Run the workflow with DBNL telemetry settings
# Note: you may have to update configuration settings based on your DBNL deployment
nat run --config_file examples/observability/simple_calculator_observability/configs/config-dbnl.yml --input "What is 1*2?"
```

As the workflow runs, telemetry data will start showing up in DBNL.

## Step 7: Analyze Traces Data in DBNL

Analyze the traces in DBNL. To analyze traces in DBNL:

1. Navigate to your DBNL deployment (e.g. http://localhost:8080/)
2. Go to Projects > nat-calculator

For additional help, see the [DBNL docs](https://docs.dbnl.com/).
