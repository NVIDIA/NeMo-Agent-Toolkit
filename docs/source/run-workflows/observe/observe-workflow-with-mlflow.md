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

# Observe a NVIDIA NeMo Agent Toolkit Workflow with MLflow

This guide shows how to send **OpenTelemetry** traces from NeMo Agent Toolkit to an [MLflow](https://mlflow.org/docs/latest/tracing/) tracking server using the built-in `mlflow` exporter (`nvidia-nat[opentelemetry]`). MLflow 3.6+ ingests OTLP spans at `<tracking-server>/v1/traces` and routes them to an experiment via the `x-mlflow-experiment-id` header. For field reference and custom OTLP endpoints, refer to [Adding Telemetry Exporters](../../extend/custom-components/telemetry-exporters.md).

## Start an MLflow tracking server

Trace ingestion requires a tracking server backed by a database store. Install MLflow (`pip install "mlflow>=3.6"`) and start it:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The OTLP ingestion endpoint is then `http://localhost:5000/v1/traces`, and the MLflow UI is at `http://localhost:5000`.

## Configure the environment

```bash
# Optional: overrides the defaults in config-mlflow.yml
export MLFLOW_OTLP_ENDPOINT="http://localhost:5000/v1/traces"
export MLFLOW_EXPERIMENT_ID="0"   # the MLflow experiment ID to route traces to (default experiment is "0")
```

## Install the OpenTelemetry extra

```bash
uv pip install -e ".[opentelemetry]"
# or, from PyPI: uv pip install "nvidia-nat[opentelemetry]"
```

## Run the simple calculator observability example

From the root of the NeMo Agent Toolkit repository:

```bash
uv pip install -e examples/observability/simple_calculator_observability/

nat run --config_file examples/observability/simple_calculator_observability/configs/config-mlflow.yml --input "What is 2 * 4?"
```

You should observe a log line such as `Started exporter 'mlflow'`. Open the MLflow UI at `http://localhost:5000`, select the experiment matching `MLFLOW_EXPERIMENT_ID`, and view the workflow trace under the **Traces** tab.

## Related configuration

- Example config: `examples/observability/simple_calculator_observability/configs/config-mlflow.yml`
- [Telemetry exporters reference](../../extend/custom-components/telemetry-exporters.md)
