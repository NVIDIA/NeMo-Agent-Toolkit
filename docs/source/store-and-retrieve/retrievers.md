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

Retrievers are an important component of Retrieval Augmented Generation (RAG) workflows which allow LLMs to search a data store for content which is semantically similar to a query, which can be used as context by the LLM when providing a response to the query. Within NeMo Agent toolkit, retrievers are a configurable component that can be used within functions, similar to LLMs and Embedders, to provide a consistent read-only interface for connecting to different data store providers.

## Features
 - **Standard Interface**: Retrievers implement a standard search interface, allowing for compatibility across different retriever implementations.
 - **Standard Output Format**: Retrievers also implement a standard output format along with conversion functions to provide retriever output as a dictionary or string.
 - **Extensible Via Plugins**: Additional retrievers can be added as plugins by developers to support more data stores.
 - **Additional Framework Implementations**: Retrievers can be loaded using a framework implementation rather than the default NeMo Agent toolkit retriever implementation.

## Included Retrievers
 - [Milvus](https://milvus.io/docs)
 - [NeMo Retriever](https://docs.nvidia.com/nemo/retriever/index.html)

## Usage
### Configuration
Retrievers are configured similarly to other NeMo Agent toolkit components, such as Functions and LLMs. Each Retriever provider (e.g., Milvus) has a Pydantic config object which defines its configurable parameters and type. These parameters can then be configured in the config file under the `retrievers` section.

Below is an example config object for the NeMo Retriever:
```python
class NemoRetrieverConfig(RetrieverBaseConfig, name="nemo_retriever"):
"""
Configuration for a Retriever which pulls data from a Nemo Retriever service.
"""
uri: HttpUrl = Field(description="The uri of the Nemo Retriever service.")
collection_name: str | None = Field(description="The name of the collection to search", default=None)
top_k: int | None = Field(description="The number of results to return", gt=0, le=50, default=None)
output_fields: list[str] | None = Field(
    default=None,
    description="A list of fields to return from the datastore. If 'None', all fields but the vector are returned.")
timeout: int = Field(default=60, description="Maximum time to wait for results to be returned from the service.")
nvidia_api_key: str | None = Field(
    description="API key used to authenticate with the service. If 'None', will use ENV Variable 'NVIDIA_API_KEY'",
    default=None,
)
```
This retriever can be easily configured in the config file such as in the below example:
```yaml
retrievers:
    my_retriever:
        _type: nemo_retriever
        uri: http://my-nemo-service-url
        collection_name: "test_collection"
        top_k: 10
```
In this example the `uri`, `collection_name`, and `top_k` are specified, while the default values for `output_fields` and `timeout` are used, and the `nvidia_api_key` will be pulled from the `NVIDIA_API_KEY` environment variable.

This configured retriever can then be used as an argument for a function which uses a retriever (such as the `aiq_retriever` function). The `aiq_retriever` function is a simple function to provide the configured retriever as an LLM tool. Its config is shown below

```python
class AIQRetrieverConfig(FunctionBaseConfig, name="aiq_retriever"):
    """
    AIQRetriever tool which provides a common interface for different vectorstores. Its
    configuration uses clients, which are the vectorstore-specific implementaiton of the retriever interface.
    """
    retriever: RetrieverRef = Field(description="The retriever instance name from the workflow configuration object.")
    raise_errors: bool = Field(
        default=True,
        description="If true the tool will raise exceptions, otherwise it will log them as warnings and return []",
    )
    topic: str = Field(default=None, description="Used to provide a more detailed tool description to the agent")
    description: str = Field(default=None, description="If present it will be used as the tool description")
```

Here is an example configuration of an `aiq_retriever` function that uses a `nemo_retriever`:
```yaml
retrievers:
    my_retriever:
        _type: nemo_retriever
        uri: http://my-nemo-service-url
        collection_name: "test_collection"
        top_k: 10

functions:
    aiq_retriever_tool:
        _type: aiq_retriever
        retriever: my_retriever
        topic: "AIQ documentation"
```

### Developing with Retrievers
Alternatively, you can use a retriever as a component in your own function, such as a custom built RAG workflow. When building a function that uses a retriever you can instantiate the retriever using the builder. Like other components, you can reference the retriever by name and specify the framework you want to use. Unlike other components, you can also omit the framework to get an instance of an `AIQRetriever`.

```python
@register_function(config_type=MyFunctionConfig)
async def my_function(config: MyFunctionConfig, builder: Builder):

    # Build an AIQRetriever
    aiq_retriever = await builder.get_retriever(config.retriever)

    # Build a langchain Retriever
    langchain_retriever = await builder.get_retriever(config.retriever, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
```

Retrievers expose a `search` method for retrieving data that takes a single required argument, "query", and any number of optional keyword arguments. NeMo Agent toolkit Retrievers support a `bind` method which can be used to set or override defaults for these optional keyword arguments. Any additional required, unbound, parameters can be inspected using the `get_unbound_params` method. This provides flexibility in how retrievers are used in functions, allowing for all search parameters to be specified in the config, or allowing some to be specified by the agent when the function is called.
