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

# LLMs

[Large language models (LLMs)](https://www.nvidia.com/en-us/glossary/large-language-models/)  are deep learning algorithms that can recognize, summarize, translate, predict, and generate content using very large datasets.

## Supported LLM Providers

NVIDIA NeMo Agent toolkit supports the following LLM providers:
| Provider | Type | Description |
|----------|------|-------------|
| [NVIDIA NIM](https://build.nvidia.com) | `nim` | NVIDIA Inference Microservice (NIM) |
| [OpenAI](https://openai.com) | `openai` | OpenAI API |
| [AWS Bedrock](https://aws.amazon.com/bedrock/) | `aws_bedrock` | AWS Bedrock API |
| [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/quickstart) | `azure_openai` | Azure OpenAI API |
| [LiteLLM](https://github.com/BerriAI/litellm) | `litellm` | LiteLLM API |
| [HuggingFace](https://huggingface.co) | `huggingface` | HuggingFace API |


## LLM Configuration

The LLM configuration is defined in the `llms` section of the workflow configuration file. The `_type` value refers to the LLM provider, and the `model_name` value always refers to the name of the model to use.

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
  openai_llm:
    _type: openai
    model_name: gpt-4o-mini
  aws_bedrock_llm:
    _type: aws_bedrock
    model_name: meta/llama-3.1-70b-instruct
    region_name: us-east-1
  azure_openai_llm:
    _type: azure_openai
    azure_deployment: gpt-4o-mini
  litellm_llm:
    _type: litellm
    model_name: gpt-4o
  huggingface_llm:
    _type: huggingface
    model_name: Qwen/Qwen3Guard-Gen-0.6B
```

### NVIDIA NIM

You can use the following environment variables to configure the NVIDIA NIM LLM provider:

* `NVIDIA_API_KEY` - The API key to access NVIDIA NIM resources


The NIM LLM provider is defined by the {py:class}`~nat.llm.nim_llm.NIMModelConfig` class.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `max_tokens` - The maximum number of tokens to generate
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request

:::{note}
`temperature` and `top_p` are model-gated fields and may not be supported by all models. If unsupported and explicitly set, validation will fail. See [Gated Fields](../../extend/custom-components/gated-fields.md) for details.
:::

### OpenAI

You can use the following environment variables to configure the OpenAI LLM provider:

* `OPENAI_API_KEY` - The API key to access OpenAI resources


The OpenAI LLM provider is defined by the {py:class}`~nat.llm.openai_llm.OpenAIModelConfig` class.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `max_tokens` - The maximum number of tokens to generate
* `seed` - The seed to use for the model
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request

:::{note}
`temperature` and `top_p` are model-gated fields and may not be supported by all models. If unsupported and explicitly set, validation will fail. See [Gated Fields](../../extend/custom-components/gated-fields.md) for details.
:::

### AWS Bedrock

The AWS Bedrock LLM provider is defined by the {py:class}`~nat.llm.aws_bedrock_llm.AWSBedrockModelConfig` class.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model. This field is ignored for LlamaIndex.
* `max_tokens` - The maximum number of tokens to generate
* `context_size` - The maximum number of tokens available for input. This is only required for LlamaIndex. This field is ignored for LangChain/LangGraph.
* `region_name` - The region to use for the model
* `base_url` - The base URL to use for the model
* `credentials_profile_name` - The credentials profile name to use for the model
* `max_retries` - The maximum number of retries for the request

### Azure OpenAI

You can use the following environment variables to configure the Azure OpenAI LLM provider:

* `AZURE_OPENAI_API_KEY` - The API key to access Azure OpenAI resources
* `AZURE_OPENAI_ENDPOINT` - The Azure OpenAI endpoint to access Azure OpenAI resources

The Azure OpenAI LLM provider is defined by the {py:class}`~nat.llm.azure_openai_llm.AzureOpenAIModelConfig` class.

* `api_key` - The API key to use for the model
* `api_version` - The API version to use for the model
* `azure_endpoint` - The Azure OpenAI endpoint to use for the model
* `azure_deployment` - The name of the Azure OpenAI deployment to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `seed` - The seed to use for the model
* `max_retries` - The maximum number of retries for the request

:::{note}
`temperature` is model-gated and may not be supported by all models. See [Gated Fields](../../extend/custom-components/gated-fields.md) for details.
:::

### LiteLLM

LiteLLM is a general purpose LLM provider that can be used with any model provider that is supported by LiteLLM.
See the [LiteLLM provider documentation](https://docs.litellm.ai/docs/providers) for more information on how to use LiteLLM.

The LiteLLM LLM provider is defined by the {py:class}`~nat.llm.litellm_llm.LiteLlmModelConfig` class.

* `model_name` - The name of the model to use (dependent on the model provider)
* `api_key` - The API key to use for the model (dependent on the model provider)
* `base_url` - The base URL to use for the model
* `seed` - The seed to use for the model
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `max_retries` - The maximum number of retries for the request

### HuggingFace

HuggingFace is a general-purpose LLM provider that can be used with any model supported by the HuggingFace API.
See the [HuggingFace documentation](https://huggingface.co/docs) for more information.

The HuggingFace LLM provider is defined by the {py:class}`~nat.llm.huggingface_llm.HuggingFaceConfig` class.

* `model_name` - The HuggingFace model name or path (for example, `Qwen/Qwen3Guard-Gen-0.6B`)
* `device` - Device for model execution: `cpu`, `cuda`, `cuda:0`, or `auto` (default: `auto`)
* `dtype` - Torch data type: `float16`, `bfloat16`, `float32`, or `auto` (default: `auto`)
* `max_new_tokens` - Maximum number of new tokens to generate (default: `128`)
* `temperature` - Sampling temperature (default: `0.0`)
* `trust_remote_code` - Whether to trust remote code when loading the model (default: `false`)

:::{note}
HuggingFace is a built-in NeMo Agent Toolkit LLM provider, but requires `nvidia-nat[huggingface]` for it to be used.
In a source installation, `uv pip install -e '.[huggingface]'` can be specified to install the required dependencies.
:::

### NVIDIA Dynamo (experimental)

Dynamo is an inference engine agnostic LLM provider designed to optimize KV cache reuse of LLMs served on NVIDIA hardware. See the [ai-dynamo repository](https://github.com/ai-dynamo/dynamo) for instructions on how to use Dynamo.

The Dynamo LLM provider is defined by the {py:class}`~nat.llm.dynamo_llm.DynamoModelConfig` class. The provider mirrors the implementation of the OpenAI provider, with additional prefix hints for Dynamo inference optimizations.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `max_tokens` - The maximum number of tokens to generate
* `seed` - The seed to use for the model
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request
* `prefix_template` - a template for conversation prefix IDs. Setting to null will disable use of `prefix_template`, `prefix_total_requests`, `prefix_osl`, and `prefix_iat`
* `prefix_total_requests` - Expected number of requests for this conversation
* `prefix_osl` - Output sequence length for the Dynamo router
* `prefix_iat` - Inter-arrival time hint for the Dynamo router
* `request_timeout` - HTTP request timeout in seconds for Dynamo LLM requests

## Testing Provider
### `nat_test_llm`
`nat_test_llm` is a development and testing provider intended for examples and CI. It is not intended for production use.

* Installation: `uv pip install nvidia-nat-test`
* Purpose: Deterministic cycling responses for quick validation
* Not for production

Minimal YAML example with `chat_completion`:

```yaml
llms:
  main:
    _type: nat_test_llm
    response_seq: [alpha, beta, gamma]
    delay_ms: 0
workflow:
  _type: chat_completion
  llm_name: main
  system_prompt: "Say only the answer."
```

* Learn how to add your own LLM provider: [Adding an LLM Provider](../../extend/custom-components/adding-an-llm-provider.md)
<!-- vale off -->
* See a short tutorial using YAML and `nat_test_llm`: [Test with nat_test_llm](../../extend/testing/test-with-nat-test-llm.md)
<!-- vale on -->

## Related Topics

```{toctree}
:titlesonly:

Using Local LLMs <./using-local-llms.md>
```
