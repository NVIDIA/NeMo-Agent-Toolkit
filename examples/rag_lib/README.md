# NVIDIA RAG Python Package Usage Guide

This guide demonstrates how to use a NAT agent with the NVIDIA RAG Python client as a tool.
# Get Started With NVIDIA RAG Blueprint

Clone the RAG repo from here: https://github.com/NVIDIA-AI-Blueprints/rag

Install the RAG Library using one of the following options:

# (Option 1) Build the wheel from source and install the Nvidia RAG wheel
uv build
uv pip install dist/nvidia_rag-2.2.1-py3-none-any.whl[all]

# (Option 2) Install the package in editable (development) mode from source
uv pip install -e .[all]

# (Option 3) Install the prebuilt wheel file from pypi. This does not require you to clone the repo.
uv pip install nvidia-rag[all]

Open the library usage guide in this notebook https://github.com/NVIDIA-AI-Blueprints/rag/blob/main/notebooks/rag_library_usage.ipynb and follow the steps to deploy your RAG server and ingest your documents (skip the installation steps as we have already installed the library)

An example file that you can ingest is provided under `nemo-agent-toolkit/examples/rag_lib/data/cuda.txt`

#### Configure Your Agent

Configure your Agent to use the Milvus collections for RAG. We have pre-configured a configuration file for you in `examples/RAG/simple_rag/configs/milvus_rag_config.yml`. You can modify this file to point to your Milvus instance and collections or add tools to your agent. The agent, by default, is a `tool_calling` agent that can be used to interact with the retriever component. The configuration file is shown below. You can also modify your agent to be another one of the NeMo Agent toolkit pre-built agent implementations such as the `react_agent`

    ```yaml
    general:
    use_uvloop: true


      functions:
      rag_tool:
        _type: rag_lib
        base_url: "http://localhost:19530"
        vdb_top_k: 20
        reranker_top_k: 10
        collection_names: ["test_library"]
        topic: Retrieve relevant documents from the database relevant to the query


    llms:
      nim_llm:
        _type: nim
        model_name: meta/llama-3.3-70b-instruct
        temperature: 0
        max_tokens: 4096
        top_p: 1


    workflow:
      _type: tool_calling_agent
      tool_names:
      - rag_tool
      verbose: true
      llm_name: nim_llm
    ```

    If you have a different Milvus instance or collection names, you can modify the `vdb_url` and the `collection_names` in the config file to point to your instance and collections. 
    You can also modify the retrieval parameters like `vdb_top_k`, ...
    You can also add additional functions as tools for your agent in the `functions` section.

#### Install the Workflow
```bash
uv pip install -e examples/rag_lib
```

#### Run the Workflow

```bash
nat run --config_file examples/rag_lib/src/rag_lib/configs/config.yml --input "How do I install CUDA"
```

The expected workflow result of running the above command is:
```console
['To install CUDA, you typically need to: \n1. Verify you have a CUDA-capable GPU and a supported version of your operating system.\n2. Download the NVIDIA CUDA Toolkit from the official NVIDIA website.\n3. Choose an installation method, such as a local repository installation or a network repository installation, depending on your system.\n4. Follow the specific instructions for your operating system, which may include installing local repository packages, enabling network repositories, or running installer scripts.\n5. Reboot your system and perform post-installation actions, such as setting up your environment and verifying the installation by running sample projects. \n\nPlease refer to the official NVIDIA CUDA documentation for detailed instructions tailored to your specific operating system and distribution.']



