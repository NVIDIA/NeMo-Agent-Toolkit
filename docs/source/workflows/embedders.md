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

# Embedders

## Supported Embedder Providers

NeMo Agent toolkit supports the following embedder providers:
| Provider | Type | Description |
|----------|------|-------------|
| [NVIDIA NIM](https://build.nvidia.com) | `nim` | NVIDIA Inference Microservice (NIM) |
| [OpenAI](https://openai.com) | `openai` | OpenAI API |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quickstart) | `azure_openai` | Azure OpenAI API |

## Embedder Configuration

The embedder configuration is defined in the `embedders` section of the workflow configuration file. The `_type` value refers to the embedder provider, and the `model_name` value always refers to the name of the model to use.

```yaml
embedders:
  nim_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5
  openai_embedder:
    _type: openai
    model_name: text-embedding-3-small
  azure_openai_embedder:
    _type: azure_openai
    azure_deployment: text-embedding-3-small
```

### NVIDIA NIM

You can use the following environment variables to configure the NVIDIA NIM embedder provider:

* `NVIDIA_API_KEY` - The API key to access NVIDIA NIM resources


The NIM embedder provider is defined by the {py:class}`~nat.embedder.nim_embedder.NIMEmbedderModelConfig` class.

* `model_name` - The name of the model to use
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request
* `truncate` - The truncation strategy to use for the model

### OpenAI

You can use the following environment variables to configure the OpenAI embedder provider:

* `OPENAI_API_KEY` - The API key to access OpenAI resources

The OpenAI embedder provider is defined by the {py:class}`~nat.embedder.openai_embedder.OpenAIEmbedderModelConfig` class.

* `model_name` - The name of the model to use
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request

### Azure OpenAI

You can use the following environment variables to configure the Azure OpenAI embedder provider:

* `AZURE_OPENAI_API_KEY` - The API key to access Azure OpenAI resources
* `AZURE_OPENAI_ENDPOINT` - The Azure OpenAI endpoint to access Azure OpenAI resources

The Azure OpenAI embedder provider is defined by the {py:class}`~nat.embedder.azure_openai_embedder.AzureOpenAIEmbedderModelConfig` class.

* `api_key` - The API key to use for the model
* `api_version` - The API version to use for the model
* `azure_endpoint` - The Azure OpenAI endpoint to use for the model
* `azure_deployment` - The name of the Azure OpenAI deployment to use
