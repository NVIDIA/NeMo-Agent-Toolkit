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

# Adding an LLM Provider to NVIDIA NeMo Agent Toolkit

In NeMo Agent toolkit the set of configuration parameters needed to interact with an LLM API (provider) is defined separately from the client which is tied to a given framework. To determine which LLM providers are included in the NeMo Agent toolkit installation, run the following command:
```bash
nat info components -t llm_provider
```

In NeMo Agent toolkit there are LLM providers, like NIM and OpenAI, and there are frameworks which need to use those providers, such as LangChain LlamaIndex with a client defined for each. To add support, we need to cover the combinations of providers to clients.

As an example, NeMo Agent toolkit contains multiple clients for interacting with the OpenAI API with different frameworks, each sharing the same provider configuration {class}`nat.llm.openai_llm.OpenAIModelConfig`. To view the full list of clients registered for the OpenAI LLM provider, run the following command:

```bash
nat info components -t llm_client -q openai
```

## Provider Types

In NeMo Agent toolkit, there are three provider types: `llm`, `embedder`, and `retreiver`. The three provider types are defined by their respective base configuration classes: {class}`nat.data_models.llm.LLMBaseConfig`, {class}`nat.data_models.embedder.EmbedderBaseConfig`, and {class}`nat.data_models.retriever.RetrieverBaseConfig`. This guide focuses on adding an LLM provider. However, the process for adding an embedder or retriever provider is similar.


## Defining an LLM Provider
The first step to adding an LLM provider is to subclass the {class}`nat.data_models.llm.LLMBaseConfig` class and add the configuration parameters needed to interact with the LLM API. Typically, this involves a `model_name` parameter and an `api_key` parameter; however, the exact parameters will depend on the API. The only requirement is a unique name for the provider.

Examine the previously mentioned {class}`nat.llm.openai_llm.OpenAIModelConfig` class:
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

## Mixins

Mixins are used to add additional fields to the provider configuration without needing to subclass or add additional fields to the provider configuration explicitly. Additionally, the toolkit can use the mixins for validation and opt-in functionality.

### RetryMixin

The {class}`nat.data_models.retry_mixin.RetryMixin` is a mixin that adds a `max_retries` field to the provider config. The `max_retries` field is an integer that specifies the maximum number of retries for the request.

```python
from nat.data_models.retry_mixin import RetryMixin

class OpenAIModelConfig(LLMBaseConfig, RetryMixin, name="openai"):
    """An OpenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: str | None = Field(default=None, description="OpenAI API key to interact with hosted model.")
    base_url: str | None = Field(default=None, description="Base url to the hosted model.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The OpenAI hosted model name.")
    seed: int | None = Field(default=None, description="Random seed to set for generation.")
```

### Model-Gated Fields

Some configuration parameters are only valid for certain models. The toolkit provides built-in mixins that automatically validate and default these parameters based on the selected model. For details on the mechanism, see [Model-Gated Configuration Fields](./model-gated-fields.md).

- `TemperatureMixin`: adds a `temperature` field in [0, 1], with a default of `0.0` when supported
- `TopPMixin`: adds a `top_p` field in [0, 1], with a default of `1.0` when supported

:::{note}
The built-in mixins may reject certain fields for models that do not support them (for example, GPT-5 models currently reject `temperature` and `top_p`). If a gated field is explicitly set on an unsupported model, validation will fail.
:::

#### TemperatureMixin

The {class}`nat.data_models.temperature_mixin.TemperatureMixin` is a mixin that adds a `temperature` field to the provider config. The `temperature` field is a float in [0, 1] that specifies the sampling temperature for the model.

```python
from nat.data_models.temperature_mixin import TemperatureMixin


class OpenAIModelConfig(LLMBaseConfig, TemperatureMixin, name="openai"):
    """An OpenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")


    api_key: str | None = Field(default=None, description="OpenAI API key to interact with hosted model.")
    base_url: str | None = Field(default=None, description="Base url to the hosted model.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The OpenAI hosted model name.")
    seed: int | None = Field(default=None, description="Random seed to set for generation.")
```

#### TopPMixin

The {class}`nat.data_models.top_p_mixin.TopPMixin` is a mixin that adds a `top_p` field to the provider config. The `top_p` field is a float in [0, 1] that specifies the top-p for distribution sampling.

```python
from nat.data_models.top_p_mixin import TopPMixin

class OpenAIModelConfig(LLMBaseConfig, TopPMixin, name="openai"):
    """An OpenAI LLM provider to be used with an LLM client."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    api_key: str | None = Field(default=None, description="OpenAI API key to interact with hosted model.")
    base_url: str | None = Field(default=None, description="Base url to the hosted model.")
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"),
                            serialization_alias="model",
                            description="The OpenAI hosted model name.")
```

### Registering the Provider
An asynchronous function decorated with {py:deco}`nat.cli.register_workflow.register_llm_provider` is used to register the provider with NeMo Agent toolkit by yielding an instance of {class}`nat.builder.llm.LLMProviderInfo`.

:::{note}
Registering an embedder or retriever provider is similar; however, the function should be decorated with  {py:deco}`nat.cli.register_workflow.register_embedder_provider` or  {py:deco}`nat.cli.register_workflow.register_retriever_provider`.
:::


The `OpenAIModelConfig` from the previous section is registered as follows:
`src/nat/llm/openai_llm.py`:
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
As previously mentioned, each LLM client is specific to both the LLM API and the framework being used. The LLM client is registered by defining an asynchronous function decorated with {py:deco}`nat.cli.register_workflow.register_llm_client`. The `register_llm_client` decorator receives two required parameters: `config_type`, which is the configuration class of the provider, and `wrapper_type`, which identifies the framework being used.

:::{note}
Registering an embedder or retriever client is similar. However, the function should be decorated with {py:deco}`nat.cli.register_workflow.register_embedder_client` or {py:deco}`nat.cli.register_workflow.register_retriever_client`.
:::

The wrapped function in turn receives two required positional arguments: an instance of the configuration class of the provider, and an instance of {class}`nat.builder.builder.Builder`. The function should then yield a client suitable for the given provider and framework. The exact type is dictated by the framework itself and not by NeMo Agent toolkit.

Since many frameworks provide clients for many of the common LLM APIs, in NeMo Agent toolkit, the client registration functions are often simple factory methods. For example, the OpenAI client registration function for LangChain is as follows:

`packages/nvidia_nat_langchain/src/nat/plugins/langchain/llm.py`:
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

## Test the Combination of LLM Provider and Client

After implementing a new LLM provider, it's important to verify that it works correctly with all existing LLM clients. This can be done by writing integration tests. Here's an example of how to test the integration between the NIM LLM provider and the LangChain framework:

```python
@pytest.mark.integration
async def test_nim_langchain_agent():
    """
    Test NIM LLM with LangChain agent. Requires NVIDIA_API_KEY to be set.
    """

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful AI assistant."), ("human", "{input}")])

    llm_config = NIMModelConfig(model_name="meta/llama-3.1-70b-instruct", temperature=0.0)

    async with WorkflowBuilder() as builder:
        await builder.add_llm("nim_llm", llm_config)
        llm = await builder.get_llm("nim_llm", wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        agent = prompt | llm

        response = await agent.ainvoke({"input": "What is 1+2?"})
        assert isinstance(response, AIMessage)
        assert response.content is not None
        assert isinstance(response.content, str)
        assert "3" in response.content.lower()
```

Note: Since this test requires an API key, it's marked with `@pytest.mark.integration` to exclude it from CI runs. However, these tests are necessary for maintaining and verifying the functionality of LLM providers and their client integrations.

## Packaging the Provider and Client

The provider and client will need to be bundled into a Python package, which in turn will be registered with NeMo Agent toolkit as a [plugin](../extend/plugins.md). In the `pyproject.toml` file of the package the `project.entry-points.'nat.components'` section, defines a Python module as the entry point of the plugin. Details on how this is defined are found in the [Entry Point](../extend/plugins.md#entry-point) section of the plugins document. By convention, the entry point module is named `register.py`, but this is not a requirement.

In the entry point module it is important that the provider is defined first followed by the client, this ensures that the provider is added to the NeMo Agent toolkit registry before the client is registered. A hypothetical `register.py` file could be defined as follows:
```python
# We need to ensure that the provider is registered prior to the client

import register_provider
import register_client
```
