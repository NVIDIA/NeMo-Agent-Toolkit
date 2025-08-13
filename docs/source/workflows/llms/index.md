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

# LLMs

## Supported LLM Providers

NeMo Agent toolkit supports the following LLM providers:
| Provider | Type | Description |
|----------|------|-------------|
| [NVIDIA NIM](https://build.nvidia.com) | `nim` | NVIDIA Inference Microservice (NIM) |
| [OpenAI](https://openai.com) | `openai` | OpenAI API |
| [AWS Bedrock](https://aws.amazon.com/bedrock/) | `aws_bedrock` | AWS Bedrock API |


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
```

### NVIDIA NIM

The NIM LLM provider is defined by the {py:class}`~nat.llm.nim_llm.NIMModelConfig` class.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `max_tokens` - The maximum number of tokens to generate
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request

### OpenAI

The OpenAI LLM provider is defined by the {py:class}`~nat.llm.openai_llm.OpenAIModelConfig` class.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `top_p` - The top-p value to use for the model
* `max_tokens` - The maximum number of tokens to generate
* `seed` - The seed to use for the model
* `api_key` - The API key to use for the model
* `base_url` - The base URL to use for the model
* `max_retries` - The maximum number of retries for the request

### AWS Bedrock

The AWS Bedrock LLM provider is defined by the {py:class}`~nat.llm.aws_bedrock_llm.AWSBedrockModelConfig` class.

* `model_name` - The name of the model to use
* `temperature` - The temperature to use for the model
* `max_tokens` - The maximum number of tokens to generate
* `context_size` - The context size to use for the model
* `region_name` - The region to use for the model
* `base_url` - The base URL to use for the model
* `credentials_profile_name` - The credentials profile name to use for the model


```{toctree}
:caption: LLMs

Using Local LLMs <./using-local-llms.md>
```
