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
```

### NVIDIA NIM
The NIM embedder provider is defined by the {py:class}`~nat.embedder.nim_embedder.NIMEmbedderModelConfig` class.

* `model_name` - The name of the model to use
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request
* `truncate` - The truncation strategy to use for the model

### OpenAI

The OpenAI embedder provider is defined by the {py:class}`~nat.embedder.openai_embedder.OpenAIEmbedderModelConfig` class.

* `model_name` - The name of the model to use
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request

