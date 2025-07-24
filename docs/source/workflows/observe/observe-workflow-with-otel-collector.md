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

# Observing a Workflow with Open Telemetry Collector

This guide shows how to stream OpenTelemetry (OTel) traces from your NeMo Agent toolkit workflows to the [generic OTel collector](https://opentelemetry.io/docs/collector/quick-start/), which in turn provides the ability to export those traces to many different places including file stores (like [S3](https://github.com/open-telemetry/opentelemetry-collector-contrib/tree/main/exporter/awss3exporter)), [Datadog](https://docs.datadoghq.com/opentelemetry/setup/collector_exporter/), and others.

In this guide, you will learn how to:

- Deploy the generic OTel collector with a configuration that saves traces to the local file system. The configuration can be modified to export to other systems.
- Configure your workflow (YAML) or Python script to send traces to the OTel collector.
- Run the workflow and view traces in the local file.

---

### Step 1. Configure and deploy the OTel Collector

1. [Configure the OTel Collector](https://opentelemetry.io/docs/collector/configuration/) using a `otlp` receiver and the exporter of your choice. For this example, create a file names `otelcollectorconfig.yaml`:

    ```yaml
    receivers:
      otlp:
        protocols:
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch:
        send_batch_size: 100
        timeout: 10s

    exporters:
      file:
        path: /tmp/llm_spans.json
        format: json

    service:
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch]
          exporters: [file]
    ```

2. [Install and run your configured OTel Collector](https://opentelemetry.io/docs/collector/installation/) noting the endpoint URL such as `http://localhost:4318`. For this example, run the OTel Collector using Docker and the configuration file from step 1:

```bash
mkdir otellogs
docker run -v $(pwd)/otelcollectorconfig.yaml:/etc/otelcol-contrib/config.yaml \
   -p 4318:4318 \
   -v $(pwd)/otellogs:/tmp/ \
   otel/opentelemetry-collector-contrib:0.128.0
```


### Step 2: Modify Workflow Configuration

Update your workflow configuration file to include the telemetry settings.

Example configuration:
```yaml
general:
  telemetry:
    tracing:
      otelcollector:
        _type: otelcollector
        # The endpoint where you have deployed the otel collector
        endpoint: http://0.0.0.0:4318/v1/traces
        project: simple_calculator
```

### Step 3. Run the workflow

```bash
# ensure you have installed aiqtoolkit with telemetry, eg uv pip install -e '.[telemetry]'
uv pip install -e examples/basic/function/simple_calculator
aiq run --config_file examples/basic/functions/simple_calculator/configs/config.yml --input "2 + 2"
```

As the workflow runs, spans are sent to the OTel Collector which in turn exports them based on the exporter you configured. In this example, you can view the exported traces in the local file:

```bash
cat otellogs/llm_spans.json
```
