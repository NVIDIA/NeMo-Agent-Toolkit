<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Embedders in NVIDIA NeMo Agent Toolkit

An embedder, or [embedding model](https://www.nvidia.com/en-us/glossary/vector-database#nv-title-fcf2efe582), is a model that transforms diverse data, such as text, images, charts, and video, into numerical vectors in a way that captures their meaning and nuance in a multidimensional vector space.

## Supported Embedder Providers

NeMo Agent Toolkit supports the following embedder providers:
| Provider | Type | Description |
|----------|------|-------------|
| [NVIDIA NIM](https://build.nvidia.com) | `nim` | NVIDIA Inference Microservice (NIM) |
| [OpenAI](https://openai.com) | `openai` | OpenAI API |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quickstart) | `azure_openai` | Azure OpenAI API |
| [Hugging Face](https://huggingface.co) | `huggingface` | Local sentence-transformers or remote Inference Endpoints (TEI) |

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

### Hugging Face

Hugging Face is an embedder provider that supports both local sentence-transformers models and remote TEI servers or Hugging Face Inference Endpoints. When `endpoint_url` is provided, embeddings are generated remotely. Otherwise, models are loaded and run locally.

You can use the following environment variables to configure the Hugging Face embedder provider:

* `HF_TOKEN` - The API token to access Hugging Face Inference resources

The Hugging Face embedder provider is defined by the {py:class}`~nat.embedder.huggingface_embedder.HuggingFaceEmbedderConfig` class.

* `model_name` - The Hugging Face model identifier (for example, `BAAI/bge-large-en-v1.5`). Required for local embeddings
* `endpoint_url` - Endpoint URL for TEI server or Hugging Face Inference Endpoint. When set, uses remote embedding
* `api_key` - The Hugging Face API token for authentication
* `timeout` - Request timeout in seconds (default: `120.0`)
* `device` - Device for local models: `cpu`, `cuda`, `mps`, or `auto` (default: `auto`)
* `normalize_embeddings` - Whether to normalize embeddings to unit length (default: `true`)
* `batch_size` - Batch size for embedding generation (default: `32`)
* `max_seq_length` - Maximum sequence length for input text
* `trust_remote_code` - Whether to trust remote code when loading models (default: `false`)

```yaml
embedders:
  # Local sentence-transformers embedder
  local_embedder:
    _type: huggingface
    model_name: sentence-transformers/all-MiniLM-L6-v2
    device: auto
    normalize_embeddings: true

  # Remote TEI or Inference Endpoint embedder
  tei_embedder:
    _type: huggingface
    endpoint_url: http://localhost:8081
    api_key: ${HF_TOKEN}
```
