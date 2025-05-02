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

# Adding an LLM Provider to NVIDIA Agent Intelligence Toolkit

In AIQ Toolkit the set of configuration parameters needed to interact with an LLM API (provider) is defined separately from the client which is tied to a given framework. To determine which LLM providers are included in the AIQ Toolkit installation, run the following command:
```bash
aiq info components -t llm_provider
```

In AIQ Toolkit there are LLM providers, like NIM and OpenAI, and there are frameworks which need to use those providers, such as LangChain LlamaIndex with a client defined for each. To add support, we need to cover the combinations of providers to clients.

As an example, AIQ Toolkit contains multiple clients for interacting with the OpenAI API with different frameworks, each sharing the same provider configuration {class}`aiq.llm.openai_llm.OpenAIModelConfig`. To view the full list of clients registered for the OpenAI LLM provider, run the following command:

```bash
aiq info components -t llm_client -q openai
```

## Provider Types

In AIQ Toolkit, there are three provider types: `llm`, `embedder`, and `retreiver`. The three provider types are defined by their respective base configuration classes: {class}`aiq.data_models.llm.LLMBaseConfig`, {class}`aiq.data_models.embedder.EmbedderBaseConfig`, and {class}`aiq.data_models.retriever.RetrieverBaseConfig`. This guide focuses on adding an LLM provider. However, the process for adding an embedder or retriever provider is similar.


## Defining an LLM Provider
The first step to adding an LLM provider is to subclass the {class}`aiq.data_models.llm.LLMBaseConfig` class and add the configuration parameters needed to interact with the LLM API. Typically, this involves a `model_name` parameter and an `api_key` parameter; however, the exact parameters will depend on the API. The only requirement is a unique name for the provider.

Examine the previously mentioned {class}`aiq.llm.openai_llm.OpenAIModelConfig` class:
```python
class OpenAIModelConfig(LLMBaseConfig, name="openai"):
    """An OpenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=())

    api_key: str | None = Field(default=None, description="OpenAI API key to interact with hosted model.")
    base_url: str | None = Field(default=None, description="Base url to the hosted model.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The OpenAI hosted model name.")
    temperature: float = Field(default=0.0, description="Sampling temperature in [0, 1].")
    top_p: float = Field(default=1.0, description="Top-p for distribution sampling.")
    seed: int | None = Field(default=None, description="Random seed to set for generation.")
    max_retries: int = Field(default=10, description="The max number of retries for the request.")
```


### Registering the Provider
An asynchronous function decorated with {py:deco}`aiq.cli.register_workflow.register_llm_provider` is used to register the provider with AIQ Toolkit by yielding an instance of {class}`aiq.builder.llm.LLMProviderInfo`.

:::{note}
Registering an embedder or retriever provider is similar; however, the function should be decorated with  {py:deco}`aiq.cli.register_workflow.register_embedder_provider` or  {py:deco}`aiq.cli.register_workflow.register_retriever_provider`.
:::


The `OpenAIModelConfig` from the previous section is registered as follows:
`src/aiq/llm/openai_llm.py`:
```python
@register_llm_provider(config_type=OpenAIModelConfig)
async def openai_llm(config: OpenAIModelConfig, builder: Builder):

    yield LLMProviderInfo(config=config, description="An OpenAI model for use with an LLM client.")
```

In the above example we didn't need to take any additional actions other than yielding the provider info. However, in some cases additional set up may be required, such as connecting to a cluster and performing validation could be performed in this method. In addition to this, any cleanup that needs to be done when the provider is no longer needed can be performed after the `yield` statement in the `finally` clause of a `try` statement. If this were needed we could update the above example as follows:
```python
@register_llm_provider(config_type=OpenAIModelConfig)
async def openai_llm(config: OpenAIModelConfig, builder: Builder):
    # Perform any setup actions here and pre-flight checks here raising an exception if needed
    try:
        yield LLMProviderInfo(config=config, description="An OpenAI model for use with an LLM client.")
    finally:
        # Perform any cleanup actions here
```

## LLM Clients
As previously mentioned, each LLM client is specific to both the LLM API and the framework being used. The LLM client is registered by defining an asynchronous function decorated with {py:deco}`aiq.cli.register_workflow.register_llm_client`. The `register_llm_client` decorator receives two required parameters: `config_type`, which is the configuration class of the provider, and `wrapper_type`, which identifies the framework being used.

:::{note}
Registering an embedder or retriever client is similar. However, the function should be decorated with {py:deco}`aiq.cli.register_workflow.register_embedder_client` or {py:deco}`aiq.cli.register_workflow.register_retriever_client`.
:::

The wrapped function in turn receives two required positional arguments: an instance of the configuration class of the provider, and an instance of {class}`aiq.builder.builder.Builder`. The function should then yield a client suitable for the given provider and framework. The exact type is dictated by the framework itself and not by AIQ Toolkit.

Since many frameworks provide clients for many of the common LLM APIs, in AIQ Toolkit, the client registration functions are often simple factory methods. For example, the OpenAI client registration function for LangChain is as follows:

`packages/aiqtoolkit_langchain/src/aiq/plugins/langchain/llm.py`:
```python
@register_llm_client(config_type=OpenAIModelConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def openai_langchain(llm_config: OpenAIModelConfig, builder: Builder):

    from langchain_openai import ChatOpenAI

    yield ChatOpenAI(**llm_config.model_dump(exclude={"type"}, by_alias=True))
```

Similar to the registration function for the provider, the client registration function can perform any necessary setup actions before yielding the client, along with cleanup actions after the `yield` statement.

:::{note}
In the above example, the `ChatOpenAI` class is imported lazily, allowing for the client to be registered without importing the client class until it is needed. Thus, improving performance and startup times.
:::

## Packaging the Provider and Client

The provider and client will need to be bundled into a Python package, which in turn will be registered with AIQ Toolkit as a [plugin](../concepts/plugins.md). In the `pyproject.toml` file of the package the `project.entry-points.'aiq.components'` section, defines a Python module as the entry point of the plugin. Details on how this is defined are found in the [Entry Point](../concepts/plugins.md#entry-point) section of the plugins document. By convention, the entry point module is named `register.py`, but this is not a requirement.

In the entry point module it is important that the provider is defined first followed by the client, this ensures that the provider is added to the AIQ Toolkit registry before the client is registered. A hypothetical `register.py` file could be defined as follows:
```python
# We need to ensure that the provider is registered prior to the client

import register_provider
import register_client
```
