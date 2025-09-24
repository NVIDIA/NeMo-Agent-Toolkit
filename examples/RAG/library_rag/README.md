# NVIDIA RAG Python Package Usage Guide

This guide demonstrates how to use a NAT agent with the NVIDIA RAG Python client as a tool.
## Table of Contents

- [Installation](#installation)
- [Setup Dependencies](#setup-dependencies)
- [API Usage Examples](#api-usage-examples)
- [Collection Management](#collection-management)
- [Document Operations](#document-operations)
- [RAG Queries](#rag-queries)
- [Search Operations](#search-operations)
- [Advanced Features](#advanced-features)
- [Cleanup Operations](#cleanup-operations)

## Installation

> **Note**: Python version **3.12 or higher** is supported.

### Prerequisites

1. **Install Python >= 3.12 and development headers:**
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.12
   sudo apt-get install python3.12-dev
   ```

2. **Install uv:**
   Follow instructions from [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

3. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   uv venv --python=python3.12
   
   # Activate virtual environment
   source .venv/bin/activate
   ```

### Installation 

```bash
uv pip install nvidia-rag[all]
```

### Verify Installation

Check that the package is installed in your virtual environment:

```bash
uv pip show nvidia_rag | grep Location
```

The location should be inside your virtual environment at: `<workspace_path>/rag/.venv/lib/python3.12/site-packages`


## Setup Dependencies

### Prerequisites

Fulfill the [prerequisites](../docs/quickstart.md#prerequisites) to setup Docker on your system.

### 1. Configure API Key

First, obtain an NGC API key by following the steps [here](../docs/quickstart.md#obtain-an-api-key).

```python
import os
from getpass import getpass
from dotenv import load_dotenv

# Set your NGC API key
if not os.environ.get("NGC_API_KEY", "").startswith("nvapi-"):
    candidate_api_key = getpass("NVAPI Key (starts with nvapi-): ")
    assert candidate_api_key.startswith("nvapi-"), f"{candidate_api_key[:5]}... is not a valid key"
    os.environ["NGC_API_KEY"] = candidate_api_key
```

### 2. Docker Login

```bash
echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

### 3. Load Default Configuration

```python
load_dotenv(dotenv_path=".env_library", override=True)
```

> **💡 Tip:** Override default configurations using `os.environ` in your code. Reimport the `nvidia_rag` package and restart the Nvidia Ingest runtime for changes to take effect.

### 4. Setup Milvus Vector Database

Configure GPU device (default uses GPU indexing):

```python
os.environ["VECTORSTORE_GPU_DEVICE_ID"] = "0"
```

> **Note:** For CPU-only Milvus, follow instructions in [milvus-configuration.md](../docs/milvus-configuration.md).

Start Milvus:
```bash
docker compose -f ../deploy/compose/vectordb.yaml up -d
```

### 5. Setup NIMs (Neural Inference Microservices)

Choose either on-premises or cloud-hosted models:

#### Option 1: On-Premises Models

Ensure you meet the [hardware requirements](../README.md#hardware-requirements). Default configuration requires 2xH100.

```bash
# Create model cache directory
mkdir -p ~/.cache/model-cache
```

```python
# Configure model directory
os.environ["MODEL_DIRECTORY"] = os.path.expanduser("~/.cache/model-cache")

# Configure GPU IDs for microservices
os.environ["EMBEDDING_MS_GPU_ID"] = "0"
os.environ["RANKING_MS_GPU_ID"] = "0" 
os.environ["YOLOX_MS_GPU_ID"] = "0"
os.environ["YOLOX_GRAPHICS_MS_GPU_ID"] = "0"
os.environ["YOLOX_TABLE_MS_GPU_ID"] = "0"
os.environ["OCR_MS_GPU_ID"] = "0"
os.environ["LLM_MS_GPU_ID"] = "1"
```

Deploy NIMs (may take time for model downloads):
```bash
USERID=$(id -u) docker compose -f ../deploy/compose/nims.yaml up -d
```

Monitor container status:
```bash
docker ps
```

Ensure all containers are running and healthy:
- nemoretriever-ranking-ms (healthy)
- compose-page-elements-1
- compose-paddle-1  
- compose-graphic-elements-1
- compose-table-structure-1
- nemoretriever-embedding-ms (healthy)
- nim-llm-ms (healthy)

#### Option 2: NVIDIA Cloud Models

```python
os.environ["APP_LLM_MODELNAME"] = "nvidia/llama-3_3-nemotron-super-49b-v1_5"
os.environ["APP_EMBEDDINGS_MODELNAME"] = "nvidia/llama-3.2-nv-embedqa-1b-v2"
os.environ["APP_RANKING_MODELNAME"] = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
os.environ["APP_EMBEDDINGS_SERVERURL"] = ""
os.environ["APP_LLM_SERVERURL"] = ""
os.environ["APP_RANKING_SERVERURL"] = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking/v1"
os.environ["EMBEDDING_NIM_ENDPOINT"] = "https://integrate.api.nvidia.com/v1"
os.environ["OCR_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/cv/baidu/paddleocr"
os.environ["OCR_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
os.environ["YOLOX_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"
os.environ["YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL"] = "http"
os.environ["YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT"] = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
os.environ["YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL"] = "http"
```

### 6. Setup NVIDIA Ingest Runtime

```bash
docker compose -f ../deploy/compose/docker-compose-ingestor-server.yaml up nv-ingest-ms-runtime redis -d
```

Open the RAG Playground at localhost:3080, create a new collection and save it. or you can use the API for that, see API Usage examples below:

## API Usage Examples

### Setup Logging

```python
import logging

LOGLEVEL = logging.WARNING  # Set to INFO, DEBUG, WARNING or ERROR
logging.basicConfig(level=LOGLEVEL)

for name in logging.root.manager.loggerDict:
    if name == "nvidia_rag" or name.startswith("nvidia_rag."):
        logging.getLogger(name).setLevel(LOGLEVEL)
    if name == "nv_ingest_client" or name.startswith("nv_ingest_client."):
        logging.getLogger(name).setLevel(LOGLEVEL)
```

### Import Packages

```python
from nvidia_rag import NvidiaRAG, NvidiaRAGIngestor

rag = NvidiaRAG()
ingestor = NvidiaRAGIngestor()
```

## Collection Management

### Create a New Collection

```python
response = ingestor.create_collection(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
    # Optional: Create collection with metadata schema
    # metadata_schema = [
    #     {
    #         "name": "meta_field_1",
    #         "type": "string", 
    #         "description": "Description field for the document"
    #     }
    # ]
)
print(response)
```

### List All Collections

```python
response = ingestor.get_collections(vdb_endpoint="http://localhost:19530")
print(response)
```

### Delete Collections

```python
response = ingestor.delete_collections(
    vdb_endpoint="http://localhost:19530", 
    collection_names=["test_library"]
)
print(response)
```

## Document Operations

### Upload Documents

```python
response = await ingestor.upload_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
    blocking=False,
    split_options={"chunk_size": 512, "chunk_overlap": 150},
    filepaths=[
        "../data/multimodal/woods_frost.docx",
        "../data/multimodal/multimodal_test.pdf",
    ],
    generate_summary=False,
    # Optional: Add custom metadata
    # custom_metadata=[
    #     {
    #         "filename": "multimodal_test.pdf",
    #         "metadata": {"meta_field_1": "multimodal document 1"}
    #     },
    #     {
    #         "filename": "woods_frost.docx", 
    #         "metadata": {"meta_field_1": "multimodal document 2"}
    #     }
    # ]
)
print(response)
```

### Check Upload Status

```python
response = await ingestor.status(task_id="YOUR_TASK_ID_HERE")
print(response)
```

### Update Documents

```python
response = await ingestor.update_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530", 
    blocking=False,
    filepaths=["../data/multimodal/woods_frost.docx"],
    generate_summary=False,
)
print(response)
```

### List Documents in Collection

```python
response = ingestor.get_documents(
    collection_name="test_library",
    vdb_endpoint="http://localhost:19530",
)
print(response)
```

### Delete Documents

```python
response = ingestor.delete_documents(
    collection_name="test_library",
    document_names=["../data/multimodal/multimodal_test.pdf"],
    vdb_endpoint="http://localhost:19530",
)
print(response)
```


#### Configure Your Agent

Configure your Agent to use the Milvus collections for RAG. We have pre-configured a configuration file for you in `examples/RAG/simple_rag/configs/milvus_rag_config.yml`. You can modify this file to point to your Milvus instance and collections or add tools to your agent. The agent, by default, is a `tool_calling` agent that can be used to interact with the retriever component. The configuration file is shown below. You can also modify your agent to be another one of the NeMo Agent toolkit pre-built agent implementations such as the `react_agent`

    ```yaml
    general:
  use_uvloop: true


functions:
  library_rag_tool:
    _type: library_rag
    base_url: "http://localhost:8081"
    reranker_top_k: 2
    vdb_top_k: 10
    vdb_endpoint: "http://milvus:19530"
    collection_names: ["cuda"]
    enable_query_rewriting: True
    enable_reranker: True

    #description: Retrieve documents given the input query

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
   - library_rag_tool
  llm_name: nim_llm
  verbose: true
    ```

    If you have a different Milvus instance or collection names, you can modify the `vdb_url` and the `collection_names` in the config file to point to your instance and collections. 
    You can also modify the retrieval parameters like `vdb_top_k`, ...
    You can also add additional functions as tools for your agent in the `functions` section.

#### Run the Workflow

```bash
nat run --config_file examples/RAG/library_rag/configs/config.yml --input "How do I install CUDA"
```

The expected workflow result of running the above command is:
```console
['To install CUDA, you typically need to: \n1. Verify you have a CUDA-capable GPU and a supported version of your operating system.\n2. Download the NVIDIA CUDA Toolkit from the official NVIDIA website.\n3. Choose an installation method, such as a local repository installation or a network repository installation, depending on your system.\n4. Follow the specific instructions for your operating system, which may include installing local repository packages, enabling network repositories, or running installer scripts.\n5. Reboot your system and perform post-installation actions, such as setting up your environment and verifying the installation by running sample projects. \n\nPlease refer to the official NVIDIA CUDA documentation for detailed instructions tailored to your specific operating system and distribution.']



