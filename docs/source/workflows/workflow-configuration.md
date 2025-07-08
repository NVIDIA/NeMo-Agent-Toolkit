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

# Workflow Configuration

AIQ toolkit workflows are defined by a [YAML configuration file](#workflow-configuration-file), which specifies which entities (functions, LLMs, embedders, etc.) to use in the workflow, along with general configuration settings.

The configuration attributes of each entity in AIQ toolkit is defined by a [Configuration Object](#configuration-object). This object defines both the type and optionally the default value of each attribute. Any attribute without a default value is required to be specified in the configuration file.

## Configuration Object
Each AIQ toolkit tool requires a configuration object which inherits from {py:class}`~aiq.data_models.function.FunctionBaseConfig`. The `FunctionBaseConfig` class and ultimately all AIQ toolkit configuration objects are subclasses of the [`pydantic.BaseModel `](https://docs.pydantic.dev/2.9/api/base_model/#pydantic.BaseModel) class from the [Pydantic Library](https://docs.pydantic.dev/2.9/), which provides a way to define and validate configuration objects. Each configuration object defines the parameters used to create runtime instances of functions (or other component type), each with different functionality based on configuration settings. It is possible to define nested functions that access other component runtime instances by name. These could be other `functions`, `llms`, `embedders`, `retrievers`, or `memory`. To facilitate nested runtime instance discovery, each component must be initialized in order based on the dependency tree. Enabling this feature requires configuration object parameters that refer to other component instances by name use a `ComponentRef` `dtype` that matches referred component type. The supported `ComponentRef` types are enumerated below:

- `FunctionRef`: Refers to a registered function by its instance name in the `functions` section configuration object.
- `LLMRef`: Refers to a registered LLM by its instance name in the `llms` section of the configuration object.
- `EmbedderRef`: Refers to a registered embedder by its instance name in the `embedders` section of the configuration object.
- `RetrieverRef`: Refers to a registered retriever by its instance name in the `retrievers` section of the configuration object.
- `MemoryRef`: Refers to a registered memory by its instance name in the `memory` section of the configuration object.


## Workflow Configuration File

The workflow configuration file is a YAML file that specifies the tools and models to use in the workflow, along with general configuration settings. To illustrate how these are organized, we will examine the configuration of the simple workflow.

`examples/basic/functions/simple/configs/config.yml`:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

embedders:
  nv-embedqa-e5-v5:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5

workflow:
  _type: react_agent
  tool_names: [webpage_query, current_datetime]
  llm_name: nim_llm
  verbose: true
  retry_parsing_errors: true
  max_retries: 3
```

From the above we see that it is divided into four sections: `functions`, `llms`, `embedders`, and `workflow`. There are additional optional sections not used in the above example they are: `general`, `memory`, `retrievers`, and `eval`.

### `functions`
The `functions` section contains the tools used in the workflow, in our example we have two `webpage_query` and `current_datetime`. By convention, the key matches the `_type` value, however this is not a strict requirement, and can be used to include multiple instances of the same tool.


### `llms`
This section contains the models used in the workflow. The `_type` value refers to the API hosting the model, in this case `nim` refers to an NIM model hosted on [`build.nvidia.com`](https://build.nvidia.com). The supported API types are `nim`, and `openai`.

The `model_name` value then needs to match a model hosted by the API, in our example we are using the [`meta/llama-3.1-70b-instruct`](https://build.nvidia.com/meta/llama-3_1-70b-instruct) model.

Both the `nim` and `openai` APIs support API specific attributes. For `nim` these are defined in the {py:class}`~aiq.llm.nim_llm.NIMModelConfig` class, and for `openai` these are defined in the {py:class}`~aiq.llm.openai_llm.OpenAIModelConfig` class.

### `embedders`
This section follows a the same structure as the `llms` section and serves as a way to separate the embedding models from the LLM models. In our example, we are using the [`nvidia/nv-embedqa-e5-v5`](https://build.nvidia.com/nvidia/nv-embedqa-e5-v5) model.

### `workflow`

This section ties the previous sections together by defining the tools and LLM models to use. The `tool_names` section lists the tool names from the `functions` section, while the `llm_name` section specifies the LLM model to use.

The `_type` value refers to the workflow type, in our example we are using a `react_agent` workflow. You can also use the workflow type, `tool_calling_agent`. The parameters for each are specified by the {py:class}`~aiq.agent.react_agent.register.ReActAgentWorkflowConfig` and {py:class}`~aiq.agent.tool_calling_agent.register.ToolCallAgentWorkflowConfig` classes respectively.

### `general`
This section contains general configuration settings for AngentIQ which are not specific to any workflow. The parameters for this section are specified by the {py:class}`~aiq.data_models.config.GeneralConfig` class.

:::{note}
The `use_uvloop` parameter which specifies whether to use the [`uvloop`](https://github.com/MagicStack/uvloop) event loop. This is set to `true` by default, and can provide a significant speedup in some cases, however this can also make it difficult to debug workflow issues. For debugging purposes it is recommended to set this to `false`:

```yaml
general:
  use_uvloop: false
```
:::

### `eval`
This section contains the evaluation settings for the workflow. Refer to [Evaluating AIQ toolkit Workflows](../workflows/evaluate.md) for more information.

### `memory`

This section configures integration of memory layers with tools such as the [Mem0 Platform](https://mem0.ai/). It follows the same format as the `llms` section. Refer to the [Memory Module](../store-and-retrieve/memory.md) document for an example on how this is used.

### `retrievers`

This section configures retrievers for vector stores. It follows the same format as the `llms` section. Refer to the `examples/rag_demo` example workflow for an example on how this is used.

### Environment Variable Interpolation

AIQ toolkit supports environment variable interpolation in YAML configuration files using the format `${VAR:-default_value}`. This allows you to:

1. Reference environment variables in your configuration
2. Provide default values if the environment variable is not set
3. Use empty strings as default values if needed

To illustrate this concept, an example from the `llms` section of the configuration file is provided below.

```yaml
llms:
  nim_llm:
    _type: nim
    base_url: ${NIM_BASE_URL:-"http://default.com"}  # Optional with default value
    api_key: ${NIM_API_KEY}  # Will use empty string if `NIM_API_KEY` not set
    model_name: ${MODEL_NAME:-}  # Will use empty string if `MODEL_NAME` not set
    temperature: 0.0
```

The environment variable interpolation process follow the rules enumerated below.

- `${VAR}` - Uses the value of environment variable `VAR`, or empty string if not set
- `${VAR:-default}` - Uses the value of environment variable `VAR`, or `default` if not set
- `${VAR:-}` - Uses the value of environment variable `VAR`, or empty string if not set
