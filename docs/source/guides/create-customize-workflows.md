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

# Create and Customize NVIDIA AgentIQ Workflows

Workflows are the heart of AgentIQ because they define which agentic tools and models are used to perform a given task or series of tasks. This document will walk through the process of running an existing workflow, customizing an existing workflow, adding tools to a workflow, creating a new tool, and creating a new workflows.

## Prerequisites

1. Set up your environment by following the instructions in the [Install From Source](../intro/install.md#install-from-source) section of the install guide.
1. Install AgentIQ and the AgentIQ Simple example workflow.
    ```bash
    uv pip install -e .
    uv pip install -e examples/simple
    ```

## Running a Workflow

A workflow is defined by a YAML configuration file that specifies the tools and models to use. AgentIQ provides the following ways to run a workflow:
- Using the `aiq run` command.
   - This is the simplest and most common way to run a workflow.
- Using the `aiq serve` command.
   - This starts a web server that listens for incoming requests and runs the specified workflow.
- Using the `aiq eval` command.
   - In addition to running the workflow, it also evaluates the accuracy of the workflow.
- Using the Python API
   - This is the most flexible way to run a workflow.

![Running Workflows](../_static/running_workflows.png)

### Using the `aiq run` Command
The `aiq run` command is the simplest way to run a workflow. `aiq run` receives a configuration file as specified by the `--config_file` flag, along with input that can be specified either directly with the `--input` flag or by providing a file path with the `--input_file` flag.

A typical invocation of the `aiq run` command follows this pattern:
```
aiq run --config_file <path/to/config.yml> [--input "question?" | --input_file <path/to/input.txt>]
```

The following command runs the `examples/simple` workflow with a single input question "What is LangSmith?":
```bash
aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"
```

The following command runs the same workflow with the input question provided in a file:
```bash
echo "What is LangSmith?" > .tmp/input.txt
aiq run --config_file examples/simple/configs/config.yml --input_file .tmp/input.txt
```

### Using the `aiq eval` Command
The `aiq eval` command is similar to the `aiq run` command, however in addition to running the workflow it also evaluates the accuracy of the workflow, refer to [Evaluating AgentIQ Workflows](../guides/evaluate.md) for more information.

### Using the `aiq serve` Command
The `aiq serve` command starts a web server that listens for incoming requests and runs the specified workflow. The server can be accessed with a web browser or by sending a POST request to the server's endpoint. Similar to the `aiq run` command, the `aiq serve` command requires a configuration file specified by the `--config_file` flag.

The following command runs the `examples/simple` workflow on a web server listening to the default port `8000` and default endpoint of `/generate`:
```bash
aiq serve --config_file examples/simple/configs/config.yml
```

In a separate terminal, run the following command to send a POST request to the server:
```bash
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "What is LangSmith?"
}'
```

Refer to `aiq serve --help` for more information on how to customize the server.

### Using the Python API

Using the Python API for running workflows is outside the scope of this document. Refer to the Python API documentation for the {py:class}`~aiq.runtime.runner.AIQRunner` class for more information.

## Understanding the Workflow Configuration File

The workflow configuration file is a YAML file that specifies the tools and models to use in a workflow, along with general configuration settings. To illustrate how these are organized, we will examine the configuration of the simple workflow that we used in the previous section.

`examples/simple/configs/config.yml`:
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

In the previous example, note that it is divided into four sections: `functions`, `llms`, `embedders`, and `workflow`. The `functions` section contains the tools used in the workflow, while `llms` and `embedders` define the models used in the workflow, and lastly the `workflow` section ties defines the workflow itself.

In the example workflow the `webpage_query` tool is used to query the LangSmith User Guide, and the `current_datetime` tool is used to get the current date and time. The questions we have asked the workflow have not involved time and the workflow would still run without the `current_datetime` tool.

The `description` entry is what is used to instruct the LLM when and how to use the tool. In this case, we explicitly defined the `description` for the `webpage_query` tool.

The `webpage_query` tool makes use of the `nv-embedqa-e5-v5` embedder, which is defined in the `embedders` section.

For details on workflow configuration, including sections not utilized in the above example, refer to the [Workflow Configuration](../concepts/workflow-configuration.md) document.

## Customizing a Workflow

In the previous sections, we have been looking at the `examples/simple` workflow that contains two tools: one that queries the LangSmith User Guide, and another that returns the current date and time. It also contains two models: an embedding model and an LLM model. After running the workflow, we can then ask it questions about LangSmith. In this section, we will discuss how to customize this workflow.

Each workflow YAML contains several configuration parameters that can be modified to customize the workflow. While copying and modifying the original, you can use the workflow YAML, which is not always necessary as some parameters can be overridden using the `--override` flag.

Examining the `examples/simple/configs/config.yml` file, the `llms` section is as follows:
```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
```

To override the `temperature` parameter for the `nim_llm`, the following command can be used:
```bash
aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"  \
  --override llms.nim_llm.temperature 0.7
```

When successful, the output contains the following line:
```
aiq.cli.cli_utils.config_override - INFO - Successfully set override for llms.nim_llm.temperature with value: 0.7
```

The `--override` flag can be specified multiple times, allowing the ability to override multiple parameters. For example, the `llama-3.1-70b-instruct` model can be replaced with the `llama-3.3-70b-instruct` using:
```bash
aiq run --config_file examples/simple/configs/config.yml --input "What is LangSmith?"  \
  --override llms.nim_llm.temperature 0.7 \
  --override llms.nim_llm.model_name meta/llama-3.3-70b-instruct
```

:::{note}
Not all parameters are specified in the workflow YAML. For each tool, there are potentially multiple optional parameters with default values that can be overridden. The `aiq info components` command can be used to list all available parameters. In this case, to list all available parameters for the LLM `nim` type run:
```bash
aiq info components -t llm_provider -q nim
```
:::

## Adding Tools to a Workflow

In the previous section, we discussed how to customize a workflow by overriding parameters. In this section, we will discuss how to add new tools to a workflow. Adding a new tool to a workflow requires copying and modifying the workflow configuration file, which, in effect, creates a new customized workflow.

AgentIQ includes several built-in tools (functions) that can be used in any workflow. To query for a list of installed tools, run the following command:
```bash
aiq info components -t function
```

The current workflow defines a tool to query the [LangSmith User Guide](https://docs.smith.langchain.com). This is defined in the `tools` section of the configuration file:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
```

However, the workflow is unaware of some related technologies, such as LangGraph, if we were to run:
```bash
aiq run --config_file examples/simple/configs/config.yml --input "How does LangSmith interact with tools like LangGraph?"
```

We would receive output similar to the following:
```
Workflow Result:
["Unfortunately, I couldn't find any information about LangSmith's interaction with LangGraph. The user guide does not mention LangGraph, and I couldn't find any relevant information through the webpage queries."]
```

We can easily solve this by updating the workflow to also query the [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction) guide.

To do this, create a copy of the original workflow configuration file. To add the LangGraph query tool to the workflow, update the YAML file updating the `functions` section from:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
```

to:
```yaml
functions:
  langsmith_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  langgraph_query:
    _type: webpage_query
    webpage_url: https://langchain-ai.github.io/langgraph/tutorials/introduction
    description: "Search for information about LangGraph. For any questions about LangGraph, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
```

Since we now have two instances of the `webpage_query` tool, we needed to update the name of the first tool to `langsmith_query`.

Finally, we need to update the `workflow.tool_names` section to include the new tool from:
```yaml
workflow:
  _type: react_agent
  tool_names: [webpage_query, current_datetime]
```

to:
```yaml
workflow:
  _type: react_agent
  tool_names: [langsmith_query, langgraph_query, current_datetime]
```

:::{note}
The resulting YAML is located at `examples/documentation_guides/workflows/custom_workflow/custom_config.yml` in the AgentIQ repository.
:::

When we rerun the workflow with the updated configuration file:
```bash
aiq run --config_file examples/documentation_guides/workflows/custom_workflow/custom_config.yml \
  --input "How does LangSmith interact with tools like LangGraph?"
```

We should receive output similar to:
```
Workflow Result:
['LangSmith interacts with LangGraph as part of an out-of-the-box solution for building complex, production-ready features with LLMs. LangGraph works in conjunction with LangSmith to provide this solution, and they are both part of the LangChain ecosystem.']
```

### Alternate Method Using a Web Search Tool
Adding individual web pages to a workflow can be cumbersome, especially when dealing with multiple web pages. An alternative method is to use a web search tool. One of the tools available in AgentIQ is the `tavily_internet_search` tool, which utilizes the [Tavily Search API](https://tavily.com/).

The `tavily_internet_search` tool is part of the `agentiq[langchain]` package, to install the package run:
```bash
# local package install from source
uv pip install -e '.[langchain]'
```

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com``](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

We will now update the `functions` section of the configuration file replacing the two `webpage_query` tools with a single `tavily_internet_search` tool entry:
```yaml
functions:
  internet_search:
    _type: tavily_internet_search
  current_datetime:
    _type: current_datetime
```

Next, we update the `workflow.tool_names` section to include the new tool:
```yaml
workflow:
  _type: react_agent
  tool_names: [internet_search, current_datetime]
```

The resulting configuration file is located at `examples/documentation_guides/workflows/custom_workflow/search_config.yml` in the AgentIQ repository.

When we re-run the workflow with the updated configuration file:
```bash
aiq run --config_file examples/documentation_guides/workflows/custom_workflow/search_config.yml \
  --input "How does LangSmith interact with tools like LangGraph?"
```

Which will then yield a slightly different result to the same question:
```
Workflow Result:
['LangSmith interacts with LangGraph through the LangChain ecosystem, which provides the foundation for building LLM applications. LangGraph provides real-time monitoring, tracing, and debugging capabilities, and it can be used in conjunction with LangSmith to build robust agentic applications.']
```

## Creating a New Tool and Workflow

In the previous examples, we have been primarily utilizing tools that were able to ingest data from web pages and perform internet searches. However, we may want to create a tool that can ingest data from local files stored on disk.

For this purpose, we will create a new empty tool using the `aiq workflow create` command. This command automates the setup process by generating the necessary files and directory structure for your new workflow.
```bash
aiq workflow create --workflow-dir examples text_file_ingest
```

This command does the following:
- Creates a new directory, `examples/text_file_ingest`.
- Sets up the necessary files and folders.
- Installs the new Python package for your workflow.

:::{note}
Due to the fact that the `aiq workflow create` command installs the new Python package, if you wish to delete the tool you will need to run the following command:
```bash
aiq workflow delete text_file_ingest
```
:::

Each workflow created in this way also creates a Python project, and by default, this will also install the project into the environment. If you want to avoid installing it into the environment you can use the `--no-install` flag.

This creates a new directory `examples/text_file_ingest` with the following layout:
```
examples/
└── text_file_ingest/
    ├── pyproject.toml
    └── src/
        └── text_file_ingest/
            ├── configs
            │   └── config.yml
            ├── __init__.py
            ├── register.py
            └── text_file_ingest_function.py
```

:::{note}
The completed code for this example can be found in the `examples/documentation_guides/workflows/text_file_ingest` directory of the AgentIQ repository.
:::

By convention, tool implementations are defined within or imported into the `register.py` file. In this example, the tool implementation exists within the `text_file_ingest_function.py` file and is imported into the `register.py` file. The `pyproject.toml` file contains the package metadata and dependencies for the tool. The `text_file_ingest_function.py` that was created for us will contain a configuration object (`TextFileIngestFunctionConfig`) along with the tool function (`text_file_ingest_function`). The next two sections will walk through customizing these.

In addition, many of these tools contain an associated workflow configuration file stored in a `config` directory, along with example data stored in a `data` directory. Since these tools are installable Python packages, and we want the workflow configuration file and data to be included in the package, they need to be located under the `examples/text_file_ingest/src/text_file_ingest` directory. For convenience symlinks can be created at the root of the project directory pointing to the actual directories. Lastly, the `README.md` file is often included in the root of the project. Resulting in a directory structure similar to the following:
```
examples/
└── text_file_ingest/
    ├── config -> src/text_file_ingest/configs
    |── data   -> src/text_file_ingest/data
    ├── pyproject.toml
    └── src/
        └── text_file_ingest/
            ├── __init__.py
            ├── configs/
            |   └── config.yml
            ├── data/
            ├── register.py
            └── text_file_ingest_function.py
```


### Customizing the Configuration Object
Given that the purpose of this tool will be similar to that of the `webpage_query` tool, we can use it as a reference and starting point. Examining the `webpage_query` tool configuration object from `examples/simple/src/aiq_simple/register.py`:
```python
class WebQueryToolConfig(FunctionBaseConfig, name="webpage_query"):
    webpage_url: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"
```

Along with renaming the class and changing the `name`, the only other configuration attribute that needs to change is replacing `webpage_url` with a glob pattern. The resulting new tool configuration object will look like:
```python
class TextFileIngestToolConfig(FunctionBaseConfig, name="text_file_ingest"):
    ingest_glob: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"
```

:::{note}
The `name` parameter; the value of this will need to match the `_type` value in the workflow configuration file.
For more details on AgentIQ configuration objects, refer to the [Configuration Object Details](../concepts/workflow-configuration.md#configuration-object) section of the [Workflow Configuration](../concepts/workflow-configuration.md) document.
:::

### Customizing the Tool Function

The `text_file_ingest_tool` function created is already correctly associated with the `TextFileIngestToolConfig` configuration object:
```python
@register_function(config_type=TextFileIngestToolConfig)
async def text_file_ingest_tool(config: TextFileIngestToolConfig, builder: Builder):
```

Examining the `webquery_tool` function (`examples/simple/src/aiq_simple/register.py`), we see that at the heart of the tool is the [`langchain_community.document_loaders.WebBaseLoader`](https://python.langchain.com/docs/integrations/document_loaders/web_base) class.

```python
    loader = WebBaseLoader(config.webpage_url)
    docs = [document async for document in loader.alazy_load()]
```

For the new tool, instead of the `WebBaseLoader` class, use the [`langchain_community.document_loaders.DirectoryLoader`](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.directory.DirectoryLoader.html) and [`langchain_community.document_loaders.TextLoader`](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.text.TextLoader.html) classes.

```python
    (ingest_dir, ingest_glob) = os.path.split(config.ingest_glob)
    loader = DirectoryLoader(ingest_dir, glob=ingest_glob, loader_cls=TextLoader)

    docs = [document async for document in loader.alazy_load()]
```

Next, update the retrieval tool definition changing the `name` parameter to `text_file_ingest`:
```python
    retriever_tool = create_retriever_tool(
        retriever,
        "text_file_ingest",
        config.description,
    )
```

The rest of the code largely remains the same resulting in the following code, the full code of this example is located at `examples/documentation_guides/workflows/text_file_ingest/src/text_file_ingest/register.py` in the AgentIQ repository:
```python
@register_function(config_type=TextFileIngestToolConfig)
async def text_file_ingest_tool(config: TextFileIngestToolConfig, builder: Builder):

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("Ingesting documents matching for the webpage: %s", config.ingest_glob)
    (ingest_dir, ingest_glob) = os.path.split(config.ingest_glob)
    loader = DirectoryLoader(ingest_dir, glob=ingest_glob, loader_cls=TextLoader)

    docs = [document async for document in loader.alazy_load()]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.chunk_size)
    documents = text_splitter.split_documents(docs)
    vector = await FAISS.afrom_documents(documents, embeddings)

    retriever = vector.as_retriever()

    retriever_tool = create_retriever_tool(
        retriever,
        "text_file_ingest",
        config.description,
    )

    async def _inner(query: str) -> str:

        return await retriever_tool.arun(query)

    yield FunctionInfo.from_fn(_inner, description=config.description)
```

### Creating the Workflow Configuration

Starting from the `custom_config.yml` file we created in the previous section, we will replace the two `webpage_query` tools with our new `text_file_ingest` tool. As a data source we will use a collection of text files located in the `examples/docs/workflows/text_file_ingest/data` directory that describes [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html).

:::{note}
If you are following this document and building this tool from scratch, you can either copy the contents of `examples/documentation_guides/workflows/text_file_ingest/data` into `examples/text_file_ingest/src/text_file_ingest/data` or populate it with your own text files.
:::

The updated `functions` section will resemble the following:
```yaml
functions:
  doca_documents:
    _type: text_file_ingest
    ingest_glob: examples/documentation_guides/workflows/text_file_ingest/data/*.txt
    description: "Search for information about DOCA and GPUNetIO. For any questions about DOCA and GPUNetIO, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  current_datetime:
    _type: current_datetime
```

Similarly, update the `workflow.tool_names` section to include the new tool:
```yaml
workflow:
  _type: react_agent
  tool_names: [doca_documents, current_datetime]
```

The resulting YAML file is located at `examples/documentation_guides/workflows/text_file_ingest/configs/config.yml` in the AgentIQ repository.

### Understanding `pyproject.toml`

The `pyproject.toml` file defines your package metadata and dependencies. In this case, the `pyproject.toml` file that was created for us is sufficient; however, that might not always be the case. The most common need to update the `pyproject.toml` file is to add additional dependencies not included with AgentIQ.

- **Dependencies**: Ensure all required libraries are listed under `[project]`.
  In the example, the tool was created inside the AgentIQ repo and simply needed to declare a dependency on `agentiq[langchain]`. If, however, your tool is intended to be distributed independently then your tool will need to declare a dependency on the specific version of AgentIQ that it was built against. To determine the version of AgentIQ run:
  ```bash
  aiq --version
  ```

 Use the first two digits of the version number. For example if the version is `1.0.0` then the dependency would be `agentiq[langchain]~=1.0`.

  ```toml
  dependencies = [
    "agentiq[langchain]~=1.0",
    # Add any additional dependencies your workflow needs
  ]
  ```

  In this example we have been using AgentIQ with LangChain, and thus we declared our dependency on `agentiq[langchain]`, that is to say AgentIQ with the LangChain integration plugin. If however we wished to use LlamaIndex, we would declare our dependency on `agentiq[llama-index]`. This is described in more detail in [Framework Integrations](../concepts/plugins.md#framework-integrations).

  we wished to use an alternate framework or other optional dependencies

- **Entry Points**: This tells AgentIQ where to find your workflow registration.

  ```toml
  [project.entry-points.'aiq.components']
  text_file_ingest = "text_file_ingest.register"
  ```

## Rebuild with Changes
By default, the `workflow create` command will install the template workflow for you to run and test.
When you modify the newly created workflow and update dependencies or code, you need to reinstall the workflow package to ensure new dependencies are installed. To do so, enter the following command:

Example:
```bash
aiq workflow reinstall text_file_ingest
```

:::{note}
Alternatively, the workflow can be uninstalled with the following command:
```bash
aiq workflow delete text_file_ingest
```
:::

### Running the Workflow

:::{note}
The following commands reference the pre-built workflow located in `examples/docs/workflows/text_file_ingest`. If you are following this document and building this tool from the beginning, you will want to replace `examples/docs/workflows/text_file_ingest` with `examples/text_file_ingest`.
:::

After completed, install the tool into the environment:
```bash
uv pip install -e examples/documentation_guides/workflows/text_file_ingest
```

Run the workflow with the following command:
```bash
aiq run --config_file examples/documentation_guides/workflows/text_file_ingest/configs/config.yml \
   --input "What does DOCA GPUNetIO to remove the CPU from the critical path?"
```

If successful, we should receive output similar to the following:
```
Workflow Result:
['DOCA GPUNetIO removes the CPU from the critical path by providing features such as GPUDirect Async Kernel-Initiated Network (GDAKIN) communications, which allows a CUDA kernel to invoke GPUNetIO device functions to receive or send data directly, without CPU intervention. Additionally, GPUDirect RDMA enables receiving packets directly into a contiguous GPU memory area. These features enable GPU-centric solutions that bypass the CPU in the critical path.']
```
