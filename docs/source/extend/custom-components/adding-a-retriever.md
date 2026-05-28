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

# Adding a Retriever Provider
New [retrievers](../../build-workflows/retrievers.md) can be added to NVIDIA NeMo Agent Toolkit by creating a plugin. The general process is the same as for most plugins, but the retriever-specific steps are outlined here.

First, create a retriever for the provider that implements the Retriever interface:
```python
from nat.plugin_api import Document
from nat.plugin_api import Retriever
from nat.plugin_api import RetrieverOutput


class ExampleRetriever(Retriever):

    def __init__(self, client):
        self._client = client

    async def search(self, query: str, **kwargs) -> RetrieverOutput:
        result = await self._client.search(query=query, **kwargs)
        return RetrieverOutput(
            results=[
                Document(page_content=item.text, metadata=item.metadata, document_id=item.id)
                for item in result.items
            ]
        )
```

Next, create the config for the provider and register it with NeMo Agent Toolkit:

```python
from nat.plugin_api import Builder
from nat.plugin_api import RetrieverBaseConfig
from nat.plugin_api import RetrieverProviderInfo
from nat.plugin_api import register_retriever_provider
from pydantic import Field
from pydantic import HttpUrl

class ExampleRetrieverConfig(RetrieverBaseConfig, name="example_retriever"):
    """
    Configuration for a Retriever provider. The parameters will depend on the particular provider. These are examples.
    """
    uri: HttpUrl = Field(description="The URI of the retriever service.")
    collection_name: str = Field(description="The name of the collection to search")
    top_k: int = Field(description="The number of results to return", gt=0, le=50, default=5)
    output_fields: list[str] | None = Field(
        default=None,
        description="A list of fields to return from the datastore. If 'None', all fields but the vector are returned.")


@register_retriever_provider(config_type=ExampleRetrieverConfig)
async def example_retriever(retriever_config: ExampleRetrieverConfig, builder: Builder):
    yield RetrieverProviderInfo(config=retriever_config,
                                description="NeMo Agent Toolkit retriever provider for...")
```
Lastly, implement and register the retriever client:

```python
from nat.plugin_api import Builder
from nat.plugin_api import register_retriever_client

@register_retriever_client(config_type=ExampleRetrieverConfig, wrapper_type=None)
async def nemo_retriever_client(config: ExampleRetrieverConfig, builder: Builder):
    from example_plugin.client import ExampleClient
    from example_plugin.retriever import ExampleRetriever

    client = ExampleClient(
        uri=str(config.uri),
        collection_name=config.collection_name,
        top_k=config.top_k,
        output_fields=config.output_fields,
    )
    retriever = ExampleRetriever(client=client)

    yield retriever
```

You can then implement and register framework-specific clients for the retriever provider, or use the config to instantiate an existing framework implementation.
