# NVIDIA RAG Python Package Usage Guide

This guide demonstrates how to use a NAT agent with the NVIDIA RAG Python client as a tool.
# Get Started With NVIDIA RAG Blueprint

Use the following documentation to get started with the NVIDIA RAG Blueprint.

- [Obtain an API Key](#obtain-an-api-key)
- [Interact using native python APIs](#interact-using-native-python-apis)
- [Deploy With Docker Compose](#deploy-with-docker-compose)
- [Deploy With Helm Chart](#deploy-with-helm-chart)
- [Data Ingestion](#data-ingestion)


## Obtain an API Key

You need to generate an API key
to access NIM services, to access models hosted in the NVIDIA API Catalog, and to download models on-premises.
For more information, refer to [NGC API Keys](https://docs.nvidia.com/ngc/gpu-cloud/ngc-private-registry-user-guide/index.html#ngc-api-keys).

To generate an API key, use the following procedure.

1. Go to https://org.ngc.nvidia.com/setup/api-keys.
2. Click **+ Generate Personal Key**.
3. Enter a **Key Name**.
4. For **Services Included**, select **NGC Catalog** and **Public API Endpoints**.
5. Click **Generate Personal Key**.

After you generate your key, export your key as an environment variable by using the following code.

```bash
export NGC_API_KEY="<your-ngc-api-key>"
```



## Deploy With Docker Compose

Use these procedures to deploy with Docker Compose for a single node deployment. Alternatively, you can [Deploy With Helm Chart](#deploy-with-helm-chart) to deploy on a Kubernetes cluster.

Developers need to deploy ingestion services and rag services using seperate dedicated docker compose files.
For both retrieval and ingestion services, by default all the models are deployed on-prem. Follow relevant section below as per your requirement and hardware availability.

- Start the Microservices
  - [Using on-prem models](#start-using-on-prem-models)
  - [Using NVIDIA hosted models](#start-using-nvidia-hosted-models)

### Prerequisites

1. Install Docker Engine. For more information, see [Ubuntu](https://docs.docker.com/engine/install/ubuntu/).

2. Install Docker Compose. For more information, see [install the Compose plugin](https://docs.docker.com/compose/install/linux/).

   a. Ensure the Docker Compose plugin version is 2.29.1 or later.

   b. After you get the Docker Compose plugin installed, run `docker compose version` to confirm.

3. To pull images required by the blueprint from NGC, you must first authenticate Docker with nvcr.io. Use the NGC API Key you created in [Obtain an API Key](#obtain-an-api-key).

   ```bash
   export NGC_API_KEY="nvapi-..."
   echo "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

4. Some containers with are enabled with GPU acceleration, such as Milvus and NVIDIA NIMS deployed on-prem. To configure Docker for GPU-accelerated containers, [install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), the NVIDIA Container Toolkit

5. Ensure you meet [the hardware requirements if you are deploying models on-prem](./support-matrix.md).

6. Change directory to the library rag example:  ```cd examples/RAG/library_rag```


### Start using on-prem models

Use the following procedure to start all containers needed for this blueprint. This launches the ingestion services followed by the rag services and all of its dependent NIMs on-prem.

1. Fulfill the [prerequisites](#prerequisites). Ensure you meet [the hardware requirements](./support-matrix.md).

2. Create a directory to cache the models and export the path to the cache as an environment variable.

   ```bash
   mkdir -p ~/.cache/model-cache
   export MODEL_DIRECTORY=~/.cache/model-cache
   ```

3. Export all the required environment variables to use on-prem models. Ensure the section `Endpoints for using cloud NIMs` is commented in this file.

   ```bash
   source deploy/.env 
   ```

4. Start all required NIMs.

   Before running the command please ensure the GPU allocation is done appropriately in the deploy/compose/.env. You might need to override them
   for the hardware you are deploying this blueprint on. The default assumes you are deploying this on a 2XH100 environment.

   ```bash
   USERID=$(id -u) docker compose -f deploy/nims.yaml up -d
   ```

   - Wait till the `nemoretriever-ranking-ms`, `nemoretriever-embedding-ms` and `nim-llm-ms`  NIMs are in healthy state before proceeding further.

   - The nemo LLM service may take upto 30 mins to start for the first time as the model is downloaded and cached. The models are downloaded and cached in the path specified by `MODEL_DIRECTORY`. Subsequent deployments will take 2-5 mins to startup based on the GPU profile.

   - The default configuration allocates one GPU (GPU ID 1) to `nim-llm-ms` which defaults to minimum GPUs needed for H100 or B200 profile. If you are deploying the solution on A100, please allocate 2 available GPUs by exporting below env variable before launching:
     ```bash
     export LLM_MS_GPU_ID=1,2
     ```

   - To start just the NIMs specific to rag or ingestion add the `--profile rag` or `--profile ingest` flag to the command.

   - Ensure all the below are running before proceeding further

     ```bash
     watch -n 2 'docker ps --format "table {{.Names}}\t{{.Status}}"'
     ```

     ```output
        NAMES                                   STATUS

        nemoretriever-ranking-ms                Up 14 minutes (healthy)
        compose-page-elements-1                 Up 14 minutes
        compose-paddle-1                        Up 14 minutes
        compose-graphic-elements-1              Up 14 minutes
        compose-table-structure-1               Up 14 minutes
        nemoretriever-embedding-ms              Up 14 minutes (healthy)
        nim-llm-ms                              Up 14 minutes (healthy)
     ```


5. Start the vector db containers from the repo root.
   ```bash
   docker compose -f deploy/vectordb.yaml up -d
   ```

   [!TIP]
   By default GPU accelerated Milvus DB is deployed. You can choose the GPU ID to be allocated using below env variable.
   ```bash
   VECTORSTORE_GPU_DEVICE_ID=0
   ```

   For B200 and A100 GPUs, use Milvus CPU indexing due to known retrieval accuracy issues with Milvus GPU indexing and search. Export following environment variables to disable Milvus GPU ingexing and search.
   ```bash
   export APP_VECTORSTORE_ENABLEGPUSEARCH=False
   export APP_VECTORSTORE_ENABLEGPUINDEX=False
   ```

6. Start the ingestion containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

7. Start the rag containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   You can check the status of the rag-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

8. Confirm all the below mentioned containers are running.

   ```bash
   docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   NAMES                                   STATUS
   compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
   ingestor-server                         Up 5 minutes
   compose-redis-1                         Up 5 minutes
   rag-playground                          Up 9 minutes
   rag-server                              Up 9 minutes
   milvus-standalone                       Up 36 minutes
   milvus-minio                            Up 35 minutes (healthy)
   milvus-etcd                             Up 35 minutes (healthy)
   nemoretriever-ranking-ms                Up 38 minutes (healthy)
   compose-page-elements-1                 Up 38 minutes
   compose-paddle-1                        Up 38 minutes
   compose-graphic-elements-1              Up 38 minutes
   compose-table-structure-1               Up 38 minutes
   nemoretriever-embedding-ms              Up 38 minutes (healthy)
   nim-llm-ms                              Up 38 minutes (healthy)
   ```

9.  Open a web browser and access `http://localhost:8090` to use the RAG Playground. You can use the upload tab to ingest files into the server or follow [the notebooks](../notebooks/) to understand the API usage.

10. To stop all running services, after making some [customizations](#next-steps)
    ```bash
    docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
    docker compose -f deploy/compose/nims.yaml down
    docker compose -f deploy/compose/docker-compose-rag-server.yaml down
    docker compose -f deploy/compose/vectordb.yaml down
    ```

**📝 Notes:**

1. A single NVIDIA A100-80GB or H100-80GB, B200 GPU can be used to start non-LLM NIMs (nemoretriever-embedding-ms, nemoretriever-ranking-ms, and ingestion services like page-elements, paddle, graphic-elements, and table-structure) for ingestion and RAG workflows. You can control which GPU is used for each service by setting these environment variables in `deploy/compose/.env` file before launching:
   ```bash
   EMBEDDING_MS_GPU_ID=0
   RANKING_MS_GPU_ID=0
   YOLOX_MS_GPU_ID=0
   YOLOX_GRAPHICS_MS_GPU_ID=0
   YOLOX_TABLE_MS_GPU_ID=0
   PADDLE_MS_GPU_ID=0
   ```

2. If the NIMs are deployed in a different workstation or outside the nvidia-rag docker network on the same system, replace the host address of the below URLs with workstation IPs.

   ```bash
   APP_EMBEDDINGS_SERVERURL="workstation_ip:8000"
   APP_LLM_SERVERURL="workstation_ip:8000"
   APP_RANKING_SERVERURL="workstation_ip:8000"
   PADDLE_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_GRAPHIC_ELEMENTS_GRPC_ENDPOINT="workstation_ip:8001"
   YOLOX_TABLE_STRUCTURE_GRPC_ENDPOINT="workstation_ip:8001"
   ```

3. Due to react limitations, any changes made to below environment variables will require developers to rebuilt the rag containers. This will be fixed in a future release.

   ```output
   # Model name for LLM
   NEXT_PUBLIC_MODEL_NAME: ${APP_LLM_MODELNAME:-meta/llama-3.1-70b-instruct}
   # Model name for embeddings
   NEXT_PUBLIC_EMBEDDING_MODEL: ${APP_EMBEDDINGS_MODELNAME:-nvidia/llama-3.2-nv-embedqa-1b-v2}
   # Model name for reranking
   NEXT_PUBLIC_RERANKER_MODEL: ${APP_RANKING_MODELNAME:-nvidia/llama-3.2-nv-rerankqa-1b-v2}
   # URL for rag server container
   NEXT_PUBLIC_CHAT_BASE_URL: "http://rag-server:8081/v1"
   # URL for ingestor container
   NEXT_PUBLIC_VDB_BASE_URL: "http://ingestor-server:8082/v1"
   ```


### Start using nvidia hosted models

1. Verify that you meet the [prerequisites](#prerequisites).

2. Open `deploy/.env` and uncomment the section `Endpoints for using cloud NIMs`.
   Then set the environment variables by executing below command.
   ```bash
   source deploy/.env
   ```


   **📝 Note:**
   When using NVIDIA hosted endpoints, you may encounter rate limiting with larger file ingestions (>10 files).

3. Start the vector db containers from the repo root.
   ```bash
   docker compose -f deploy/vectordb.yaml up -d
   ```
   [!NOTE]
   If you don't have a GPU available, you can switch to CPU-only Milvus by following the instructions in [milvus-configuration.md](./milvus-configuration.md).

   [!TIP]
   For B200 and A100 GPUs, use Milvus CPU indexing due to known retrieval accuracy issues with Milvus GPU indexing and search. Export following environment variables to disable Milvus GPU ingexing and search.
   ```bash
   export APP_VECTORSTORE_ENABLEGPUSEARCH=False
   export APP_VECTORSTORE_ENABLEGPUINDEX=False
   ```

4. Start the ingestion containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d
   ```

   [!TIP]
   You can add a `--build` argument in case you have made some code changes or have any requirement of re-building ingestion containers from source code:

   ```bash
   docker compose -f deploy/compose/docker-compose-ingestor-server.yaml up -d --build
   ```

5. Start the rag containers from the repo root. This pulls the prebuilt containers from NGC and deploys it on your system.

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d
   ```

   [!TIP]
   You can add a `--build` argument in case you have made some code changes or have any requirement of re-building containers from source code:

   ```bash
   docker compose -f deploy/compose/docker-compose-rag-server.yaml up -d --build
   ```

   You can check the status of the rag-server and its dependencies by issuing this curl command
   ```bash
   curl -X 'GET' 'http://workstation_ip:8081/v1/health?check_dependencies=true' -H 'accept: application/json'
   ```

6. Confirm all the below mentioned containers are running.

   ```bash
   docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"
   ```

   *Example Output*

   ```output
   NAMES                                   STATUS
   compose-nv-ingest-ms-runtime-1          Up 5 minutes (healthy)
   ingestor-server                         Up 5 minutes
   compose-redis-1                         Up 5 minutes
   rag-playground                          Up 9 minutes
   rag-server                              Up 9 minutes
   milvus-standalone                       Up 36 minutes
   milvus-minio                            Up 35 minutes (healthy)
   milvus-etcd                             Up 35 minutes (healthy)
   ```

7. Open a web browser and access `http://localhost:8090` to use the RAG Playground. You can use the upload tab to ingest files into the server or follow [the notebooks](../notebooks/) to understand the API usage.

8. To stop all running services, after making some [customizations](#next-steps)
    ```bash
    docker compose -f deploy/compose/docker-compose-ingestor-server.yaml down
    docker compose -f deploy/compose/docker-compose-rag-server.yaml down
    docker compose -f deploy/compose/vectordb.yaml down

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
        "examples/RAG/libraary_rag/data/cuda.txt",
    ],
    generate_summary=False,
    
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



