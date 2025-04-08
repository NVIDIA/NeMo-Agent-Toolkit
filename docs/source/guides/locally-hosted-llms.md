# Using Local LLMs

AgentIQ has the ability to interact with locally hosted LLMs, in this guide we will demonstrate how to adapt the AgentIQ simple example (`examples/simple`) to use locally hosted LLMs using two different approaches using [NVIDIA NIM](https://docs.nvidia.com/nim/) and [vLLM](https://docs.vllm.ai/).

## Using NIM
In the AgentIQ simple example the [`meta/llama-3.1-70b-instruct`](https://build.nvidia.com/meta/llama-3_1-70b-instruct) model was used. For the purposes of this guide we will be using a smaller model, the [`qwen/qwen2_5-7b-instruct`](https://build.nvidia.com/qwen/qwen2_5-7b-instruct) which is more likely to be runnable on a local workstation.

Regardless of the model you choose, the process is the same for downloading the model's container from [build.nvidia.com](https://build.nvidia.com/). Navigate to the model you wish to run locally, if it is able to be downloaded it will be labeled with the `RUN ANYWHERE` tag, the exact commands will be specified on the `Deploy` tab for the model.

### Requirements
- An NVIDIA GPU with CUDA support (exact requirements depend on the model you are using)
- [The NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installation)
- An NVIDIA API key, refer to [Obtaining API Keys](../intro/get-started.md#obtaining-api-keys) for more information.

### Downloading the NIM Containers

Login to nvcr.io with Docker:
```
$ docker login nvcr.io
Username: $oauthtoken
Password: <PASTE_API_KEY_HERE>
```

Download the container for the LLM:
```bash
docker pull nvcr.io/nim/qwen/qwen-2.5-7b-instruct:latest
```

Download the container for the embedding Model:
```bash
docker pull nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest
```


### Running the NIM Containers
Run the LLM container listening on port 8000:
```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=~/.cache/nim
mkdir -p "$LOCAL_NIM_CACHE"
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8000:8000 \
    nvcr.io/nim/qwen/qwen-2.5-7b-instruct:latest
```

Open a new terminal and run the embedding model container, listening on port 8001:
```bash
export NGC_API_KEY=<PASTE_API_KEY_HERE>
export LOCAL_NIM_CACHE=~/.cache/nim
docker run -it --rm \
    --gpus all \
    --shm-size=16GB \
    -e NGC_API_KEY \
    -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
    -u $(id -u) \
    -p 8001:8000 \
    nvcr.io/nim/nvidia/nv-embedqa-e5-v5:latest
```

### AgentIQ Configuration
To define the pipeline configuration, we will start with the `examples/simple/configs/config.yml` file and modify it to use the locally hosted LLMs, the only changes needed are to define the `base_url` for the LLM and embedding models, along with the names of the models to use.

`examples/documentation_guides/locally_hosted_llms/nim_config.yml`:
```yaml
functions:
  webpage_query:
    _type: webpage_query
    webpage_url: https://docs.smith.langchain.com/user_guide
    description: "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
    embedder_name: nv-embedqa-e5-v5
    chunk_size: 512
  current_datetime:
    _type: current_datetime

llms:
  nim_llm:
    _type: nim
    base_url: "http://localhost:8000/v1"
    model_name: qwen/qwen-2.5-7b-instruct

embedders:
  nv-embedqa-e5-v5:
    _type: nim
    base_url: "http://localhost:8001/v1"
    model_name: nvidia/nv-embedqa-e5-v5

workflow:
  _type: react_agent
  tool_names: [webpage_query, current_datetime]
  llm_name: nim_llm
  verbose: true
  retry_parsing_errors: true
  max_retries: 3
```

### Running the AgentIQ Workflow
To run the AgentIQ workflow using the locally hosted LLMs, run the following command:
```bash
aiq run --config_file examples/documentation_guides/locally_hosted_llms/nim_config.yml --input "What is LangSmith?"
```


## Using vLLM

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```


```bash
vllm serve --task embed --override-pooler-config '{"pooling_type": "MEAN"}' --port 8001  ssmits/Qwen2-7B-Instruct-embed-base
```
