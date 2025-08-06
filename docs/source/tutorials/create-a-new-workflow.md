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


# Create a New Tool and Workflow

In the [Customizing a Workflow](./customize-a-workflow.md) and [Adding Tools to a Workflow](./create-a-new-workflow.md) tutorials, we have been primarily utilizing tools that were included with the NeMo Agent toolkit. This tutorial demonstrates how to create a new tool that can ingest data from local files stored on disk.

For this purpose, create a new empty tool using the `aiq workflow create` command. This command automates the setup process by generating the necessary files and directory structure for your new workflow.
```bash
aiq workflow create --workflow-dir examples text_file_ingest
```

This command does the following:
<!-- path-check-skip-next-line -->
- Creates a new directory, `examples/text_file_ingest`.
- Sets up the necessary files and folders.
- Installs the new Python package for your workflow.

:::{note}
Due to the fact that the `aiq workflow create` command installs the new Python package, if you wish to delete the tool you will need to run the following command:
```bash
aiq workflow delete text_file_ingest
```
:::

<!-- This section needs to be updated once #559 is completed -->
Each workflow created in this way also creates a Python project, and by default, this will also install the project into the environment. If you want to avoid installing it into the environment you can use the `--no-install` flag.

<!-- path-check-skip-next-line -->
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
The completed code for this example can be found in the `examples/documentation_guides/workflows/text_file_ingest` directory of the NeMo Agent toolkit repository.
:::

<!-- path-check-skip-next-line -->
By convention, tool implementations are defined within or imported into the `register.py` file. In this example, the tool implementation exists within the `text_file_ingest_function.py` file and is imported into the `register.py` file. The `pyproject.toml` file contains the package metadata and dependencies for the tool. The `text_file_ingest_function.py` that was created for us will contain a configuration object (`TextFileIngestFunctionConfig`) along with the tool function (`text_file_ingest_function`). The next two sections will walk through customizing these.

<!-- path-check-skip-next-line -->
Many of these tools contain an associated workflow configuration file stored in a `config` directory, along with example data stored in a `data` directory. Since these tools are installable Python packages and the workflow configuration file and data must be included in the package, they need to be located under the `examples/text_file_ingest/src/text_file_ingest` directory. For convenience, symlinks can be created at the root of the project directory pointing to the actual directories. Lastly, a `README.md` file is often included in the root of the project. Resulting in a directory structure similar to the following:
```
examples/
└── text_file_ingest/
    ├── README.md
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

<!-- Remove this once #559 is completed -->
For our purposes we will need a `data` directory, along with the above mentioned symlinks which can be created with the following commands:
```bash
mkdir examples/text_file_ingest/src/text_file_ingest/data
pushd examples/text_file_ingest
ln -s src/text_file_ingest/data
ln -s src/text_file_ingest/configs
popd
```

## Customizing the Configuration Object
Given that the purpose of this tool will be similar to that of the `webpage_query` tool, you can use it as a reference and starting point. Examining the `webpage_query` tool configuration object from `examples/getting_started/simple_web_query/src/aiq_simple_web_query/register.py`:
```python
class WebQueryToolConfig(FunctionBaseConfig, name="webpage_query"):
    webpage_url: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"
```

Along with renaming the class and changing the `name`, the only other configuration attribute that needs to change is replacing `webpage_url` with a glob pattern. The resulting new tool configuration object will look like:
```python
class TextFileIngestFunctionConfig(FunctionBaseConfig, name="text_file_ingest"):
    ingest_glob: str
    description: str
    chunk_size: int = 1024
    embedder_name: EmbedderRef = "nvidia/nv-embedqa-e5-v5"
```

:::{note}
The `name` parameter; the value of this will need to match the `_type` value in the workflow configuration file.
For more details on NeMo Agent toolkit configuration objects, refer to the [Configuration Object Details](../workflows/workflow-configuration.md#configuration-object) section of the [Workflow Configuration](../workflows/workflow-configuration.md) document.
:::

## Customizing the Tool Function

The `text_file_ingest_tool` function created is already correctly associated with the `TextFileIngestFunctionConfig` configuration object:
```python
@register_function(config_type=TextFileIngestFunctionConfig)
async def text_file_ingest_function(config: TextFileIngestFunctionConfig, builder: Builder):
```

However since we are going to make use of LangChain, we need to add the `framework_wrappers` parameter to the `register_function` decorator:
```python
@register_function(config_type=TextFileIngestFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def text_file_ingest_function(config: TextFileIngestFunctionConfig, builder: Builder):
```


Examining the `webquery_tool` function (`examples/getting_started/simple_web_query/src/aiq_simple_web_query/register.py`), you can observe that at the heart of the tool is the [`langchain_community.document_loaders.WebBaseLoader`](https://python.langchain.com/docs/integrations/document_loaders/web_base) class.

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

The rest of the code largely remains the same resulting in the following code, the full code of this example is located at `examples/documentation_guides/workflows/text_file_ingest/src/text_file_ingest/register.py` in the NeMo Agent toolkit repository:
```python
@register_function(config_type=TextFileIngestFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def text_file_ingest_function(config: TextFileIngestFunctionConfig, builder: Builder):

    from langchain.tools.retriever import create_retriever_tool
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_community.document_loaders import TextLoader
    from langchain_community.vectorstores import FAISS
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings: Embeddings = await builder.get_embedder(config.embedder_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    logger.info("Ingesting documents from: %s", config.ingest_glob)
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

## Creating the Workflow Configuration

Starting from the `custom_config.yml` file you created in the previous section, replace the two `webpage_query` tools with the new `text_file_ingest` tool. For the data source, you can use a collection of text files located in the `examples/documentation_guides/workflows/text_file_ingest/data` directory that describes [DOCA GPUNetIO](https://docs.nvidia.com/doca/sdk/doca+gpunetio/index.html).

:::{note}
<!-- path-check-skip-next-line -->
If you are following this document and building this tool from scratch, you can either copy the contents of `examples/documentation_guides/workflows/text_file_ingest/data` into `examples/text_file_ingest/src/text_file_ingest/data` or populate it with your own text files.
:::

The updated `functions` section will resemble the following:
```yaml
functions:
  doca_documents:
    _type: text_file_ingest
    ingest_glob: examples/text_file_ingest/data/*.txt
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

The resulting YAML file is located at `examples/documentation_guides/workflows/text_file_ingest/configs/config.yml` in the NeMo Agent toolkit repository.

## Understanding `pyproject.toml`

The `pyproject.toml` file defines your package metadata and dependencies. In this case, the `pyproject.toml` file that was created is sufficient; however, that might not always be the case. The most common need to update the `pyproject.toml` file is to add additional dependencies that are not included with NeMo Agent toolkit.

- **Dependencies**: Ensure all required libraries are listed under `[project]`.
  In the example, the tool was created inside the NeMo Agent toolkit repo and simply needed to declare a dependency on `aiqtoolkit[langchain]`. If, however, your tool is intended to be distributed independently then your tool will need to declare a dependency on the specific version of NeMo Agent toolkit that it was built against. To determine the version of NeMo Agent toolkit run:
  ```bash
  aiq --version
  ```

  Use the first two digits of the version number. For example, if the version is `1.1.0`, then the dependency would be `aiqtoolkit[langchain]~=1.1`.

  ```toml
  dependencies = [
    "aiqtoolkit[langchain]~=1.1",
    # Add any additional dependencies your workflow needs
  ]
  ```

  In this example, you have been using NeMo Agent toolkit with LangChain. This is why the dependency is declared on `aiqtoolkit[langchain]`, that is to say NeMo Agent toolkit with the LangChain integration plugin. If you want to use LlamaIndex, declare the dependency on `aiqtoolkit[llama-index]`. This is described in more detail in [Framework Integrations](../quick-start/installing.md#framework-integrations).

- **Version**: In this example, and in NeMo Agent toolkit in general, we use [setuptools-scm](https://setuptools-scm.readthedocs.io/en/latest/) to automatically determine the version of the package based on the Git tags. We did this by setting `dynamic = ["version"]` and declaring a build dependency on both `setuptools` and `setuptools_scm` in the `build-system` section of `pyproject.toml`:
  ```toml
  [build-system]
  requires = ["setuptools", "setuptools_scm"]
  build-backend = "setuptools.build_meta"
  ```

  In addition to this, we also need to tell `setuptools_scm` where to find the root of git repository, this can be omitted if the `pyproject.toml` file is located at the root of the repository:
  ```toml
  [tool.setuptools_scm]
  root = "../../../.."
  ```

  Alternately if we did not want to do this we would instead:
  ```toml
  [build-system]
  build-backend = "setuptools.build_meta"
  requires = ["setuptools >= 64"]

  [project]
  name = "text_file_ingest"
  version = "0.1.0"
  ```

- **Entry Points**: This tells NeMo Agent toolkit where to find your workflow registration.

  ```toml
  [project.entry-points.'aiq.components']
  text_file_ingest = "text_file_ingest.register"
  ```

## Rebuild with Changes
By default, the `workflow create` command will install the template workflow for you to run and test. When you modify the newly created workflow and update dependencies or code, you need to reinstall the workflow package to ensure new dependencies are installed. To do so, enter the following command:

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

## Running the Workflow

:::{note}
<!-- path-check-skip-next-line -->
The following commands reference the pre-built workflow located in `examples/documentation_guides/workflows/text_file_ingest`. If you are following this document and building this tool from the beginning, replace `examples/documentation_guides/workflows/text_file_ingest` with `examples/text_file_ingest`.
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

If successful, you should receive output similar to the following:
```
Workflow Result:
['DOCA GPUNetIO removes the CPU from the critical path by providing features such as GPUDirect Async Kernel-Initiated Network (GDAKIN) communications, which allows a CUDA kernel to invoke GPUNetIO device functions to receive or send data directly, without CPU intervention. Additionally, GPUDirect RDMA enables receiving packets directly into a contiguous GPU memory area. These features enable GPU-centric solutions that bypass the CPU in the critical path.']
```
