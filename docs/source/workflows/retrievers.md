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

# Retrievers

Retrievers are used to retrieve relevant documents from a vector database. See the [Store and Retrieve: Retrievers](../store-and-retrieve/retrievers.md) page for more details on retrievers.

## Supported Retriever Providers

NeMo Agent toolkit supports the following retriever providers:
| Provider | Type | Description |
|----------|------|-------------|
| [NVIDIA NIM](https://build.nvidia.com) | `nemo_retriever` | NVIDIA Inference Microservice (NIM) |
| [Milvus](https://milvus.io) | `milvus_retriever` | Milvus |

## Retriever Configuration

The retriever configuration is defined in the `retrievers` section of the workflow configuration file. The `_type` value refers to the retriever provider, and the `model_name` value always refers to the name of the model to use.

```yaml
retrievers:
  nemo_retriever:
    _type: nemo_retriever
    uri: http://localhost:8000
    collection_name: my_collection
    top_k: 10
  milvus_retriever:
    _type: milvus_retriever
    uri: http://localhost:19530
    collection_name: my_other_collection
    top_k: 10
```

### NVIDIA NIM

The NIM retriever provider is defined by the {py:class}`~nat.retriever.nemo_retriever.NemoRetrieverConfig` class.

* `uri` - The URI of the NIM retriever service.
* `collection_name` - The name of the collection to search.
* `top_k` - The number of results to return.
* `output_fields` - A list of fields to return from the data store. If `None`, all fields but the vector are returned.
* `timeout` - Maximum time to wait for results to be returned from the service.
* `nvidia_api_key` - API key used to authenticate with the service. If `None`, will use ENV Variable `NVIDIA_API_KEY`.

### Milvus

The Milvus retriever provider is defined by the {py:class}`~nat.retriever.milvus.MilvusRetrieverConfig` class.

* `uri` - The URI of the Milvus service.
* `connection_args` - Dictionary of arguments used to connect to and authenticate with the Milvus service.
* `embedding_model` - The name of the embedding model to use to generate the vector from the query.
* `collection_name` - The name of the Milvus collection to search.
* `content_field` - Name of the primary field to store or retrieve.
* `top_k` - The number of results to return.
* `output_fields` - A list of fields to return from the data store. If `None`, all fields but the vector are returned.
* `search_params` - Search parameters to use when performing vector search.
* `vector_field` - Name of the field to compare with the vector generated from the query.
* `description` - If present it will be used as the tool description.
