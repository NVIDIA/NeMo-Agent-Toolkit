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

# Observing a Workflow with NVIDIA Data Flywheel

This guide provides a step-by-step process to enable observability in a NVIDIA NeMo Agent toolkit workflow that exports runtime traces to an Elasticsearch instance that is part of the [NVIDIA Data Flywheel Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-data-flywheel). The Data Flywheel Blueprint can then leverage the traces to fine-tune and evaluate smaller models which can be deployed to replace the original model to reduce latency.

The Data Flywheel integration supports LangChain/LangGraph based workflows with `nim` and `openai` LLM providers and can be enabled with just a few lines of configuration.

## Supported Framework and Provider Combinations

The Data Flywheel integration currently supports LangChain (as used in LangChain pipelines and LangGraphs) with the following LLM providers:

- `_type: openai` - OpenAI provider
- `_type: nim` - NVIDIA NIM provider

The integration captures `LLM_START` events for completions and tool calls when using these specific combinations. Other framework and provider combinations are not currently supported.

## Step 1: Prerequisites

Before using the Data Flywheel integration, ensure you have:

- NVIDIA Data Flywheel Blueprint deployed and configured
- Valid Elasticsearch credentials (username and password)

## Step 2: Install the Data Flywheel Plugin

To install the Data Flywheel plugin, run the following:

```bash
uv pip install -e '.[data-flywheel]'
```

## Step 3: Modify Workflow Configuration

Update your workflow configuration file to include the Data Flywheel telemetry settings:

```yaml
general:
  telemetry:
    tracing:
      data_flywheel:
        _type: data_flywheel_elasticsearch
        client_id: my_nat_app
        index: flywheel
        endpoint: ${ELASTICSEARCH_ENDPOINT}
        username: elastic
        password: elastic
        batch_size: 10
```

This configuration enables exporting trace data to NVIDIA Data Flywheel via Elasticsearch.

## Configuration Parameters

The Data Flywheel integration supports the following core configuration parameters:

| Parameter | Description | Required | Example |
|-----------|-------------|----------|---------|
| `client_id` | Identifier for your NAT application to distinguish traces between deployments | Yes | `"my_nat_app"` |
| `index` | Elasticsearch index name where traces will be stored | Yes | `"flywheel"` |
| `endpoint` | Elasticsearch endpoint URL | Yes | `"https://elasticsearch.example.com:9200"` |
| `username` | Elasticsearch username for authentication | No | `"elastic"` |
| `password` | Elasticsearch password for authentication | No | `"elastic"` |
| `batch_size` | Size of batch to accumulate before exporting | No | `10` |

## Step 4: Run Your Workflow

Run your workflow using the updated configuration file:

```bash
nat run --config_file config-data-flywheel.yml --input "Your workflow input here"
```

## Step 5: Monitor Trace Export

As your workflow runs, traces will be automatically exported to Elasticsearch in batches. You can monitor the export process through the NeMo Agent toolkit logs, which will show information about successful exports and any errors.

## Step 6: Access Data in Data Flywheel

Once traces are exported to Elasticsearch, they become available in the NVIDIA Data Flywheel system for:

- LLM distillation and optimization
- Performance analysis and monitoring  
- Training smaller, more efficient models
- Runtime optimization insights

## Advanced Configuration

### Workload Scoping

The Data Flywheel integration uses workload identifiers to organize traces for targeted model optimization. Understanding how to scope your workloads correctly is crucial for effective LLM distillation.

#### Default Scoping Behavior

By default, each trace receives a Data Flywheel `workload_id` that maps to the parent NeMo Agent toolkit registered function. The combination of `client_id` and `workload_id` is used by Data Flywheel to select data as the basis for training jobs.

#### Custom Scoping with `@track_unregistered_function`

For fine-grained optimization, you can create custom workload scopes using the `@track_unregistered_function` decorator. This is useful when a single registered function contains multiple LLM invocations that would benefit from separate model optimizations.

```python
from nat.profiler.decorators.function_tracking import track_unregistered_function

@track_unregistered_function(name="document_summarizer", metadata={"task_type": "summarization"})
def summarize_document(document: str) -> str:
    return llm_client.complete(f"Summarize: {document}")

@track_unregistered_function(name="question_answerer")
def answer_question(context: str, question: str) -> str:
    return llm_client.complete(f"Context: {context}\nQuestion: {question}")
```

The decorator supports:

- `name`: Custom `workload_id` (optional, defaults to function name)
- `metadata`: Additional context for traces (optional)

## Resources

For more information about NVIDIA Data Flywheel:

- [NVIDIA Data Flywheel Blueprint](https://build.nvidia.com/nvidia/build-an-enterprise-data-flywheel)
- [NVIDIA Data Flywheel Blueprint Brev.dev Launchable](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-2wggjBvDlVp4pLQD8ytZySh5m8W)
- [NVIDIA Data Flywheel GitHub Repository](https://github.com/NVIDIA-AI-Blueprints/data-flywheel)
- [NeMo Agent toolkit Observability Guide](./index.md)
