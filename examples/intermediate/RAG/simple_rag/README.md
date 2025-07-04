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


# Simple RAG Example
This is a simple example RAG application to showcase how one can configure and use the  Retriever component. This example includes:
 - The config file to run the workflow
 - A docker compose deployment for standing up Milvus
 - A script for scraping data from URLs and storing it in Milvus

 This example is intended to be illustrative and demonstrate how someone could build a simple RAG application using the retriever component and use it with an agent without any additional code required!

## Quickstart: RAG with Milvus

### Installation and Setup
If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit, and follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key.

1. From the root directory of the AIQ toolkit library, run the following commands:
    ```bash
    uv pip install -e examples/intermediate/RAG/simple_rag
    ```

1. Start the docker compose [Skip this step if you already have Milvus running]
    ```bash
    docker compose -f examples/intermediate/RAG/simple_rag/deploy/docker-compose.yaml up -d
    ```
    > Note: It can take some time for Milvus to start up. You can check the logs with:
    ```bash
    docker compose -f examples/intermediate/RAG/simple_rag/deploy/docker-compose.yaml logs --follow
    ```
1. In a new terminal, from the root of the AIQ toolkit repository, run the provided bash script to store the data in a Milvus collection. By default the script will scrape a few pages from the CUDA documentation and store the data in a Milvus collection called `cuda_docs`. It will also pull a few pages of information about the Model Context Protocol (MCP) and store it in a collection called `mcp_docs`.

    Export your NVIDIA API key:
    ```bash
    export NVIDIA_API_KEY=<YOUR API KEY HERE>
    ```

    ```bash
    source .venv/bin/activate
    scripts/bootstrap_milvus.sh
    ```

    If Milvus is running the script should work out of the box. If you want to customize the script the arguments are shown below.
    ```bash
    python scripts/langchain_web_ingest.py --help
    ```
    ```console
    usage: langchain_web_ingest.py [-h] [--urls URLS] [--collection_name COLLECTION_NAME] [--milvus_uri MILVUS_URI] [--clean_cache]

    options:
    -h, --help            show this help message and exit
    --urls URLS           Urls to scrape for RAG context (default: ['https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html', 'https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html', 'https://docs.nvidia.com/cuda/cuda-c-
                            best-practices-guide/index.html', 'https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html'])
    --collection_name COLLECTION_NAME, -n COLLECTION_NAME
                            Collection name for the data. (default: cuda_docs)
    --milvus_uri MILVUS_URI, -u MILVUS_URI
                            Milvus host URI (default: http://localhost:19530)
    --clean_cache         If true, deletes local files (default: False)
    ```

1. Configure your Agent to use the Milvus collections for RAG. We have pre-configured a configuration file for you in `examples/intermediate/RAG/simple_rag/configs/milvus_rag_config.yml`. You can modify this file to point to your Milvus instance and collections or add tools to your agent. The agent, by default, is a `tool_calling` agent that can be used to interact with the retriever component. The configuration file is shown below. You can also modify your agent to be another one of the AIQ toolkit pre-built agent implementations such as the `react_agent`

    ```yaml
    general:
      use_uvloop: true

    retrievers:
      cuda_retriever:
        _type: milvus_retriever
        uri: http://localhost:19530
        collection_name: "cuda_docs"
        embedding_model: milvus_embedder
        top_k: 10
      mcp_retriever:
        _type: milvus_retriever
        uri: http://localhost:19530
        collection_name: "mcp_docs"
        embedding_model: milvus_embedder
        top_k: 10

    functions:
      cuda_retriever_tool:
        _type: aiq_retriever
        retriever: cuda_retriever
        topic: Retrieve documentation for NVIDIA's CUDA library
      mcp_retriever_tool:
        _type: aiq_retriever
        retriever: mcp_retriever
        topic: Retrieve information about Model Context Protocol (MCP)

    llms:
      nim_llm:
        _type: nim
        model_name: meta/llama-3.3-70b-instruct
        temperature: 0
        max_tokens: 4096
        top_p: 1

    embedders:
      milvus_embedder:
        _type: nim
        model_name: nvidia/nv-embedqa-e5-v5
        truncate: "END"

    workflow:
      _type: react_agent
      tool_names:
       - cuda_retriever_tool
         - mcp_retriever_tool
      verbose: true
      llm_name: nim_llm
    ```

    If you have a different Milvus instance or collection names, you can modify the `retrievers` section of the config file to point to your instance and collections. You can also add additional functions as tools for your agent in the `functions` section.

1. Run the workflow
    ```bash
    aiq run --config_file examples/intermediate/RAG/simple_rag/configs/milvus_rag_config.yml --input "How do I install CUDA"
    ```
   The expected output of running the above command is:
    ```console
    $ aiq run --config_file examples/intermediate/RAG/simple_rag/configs/milvus_rag_config.yml --input "How do I install CUDA"
    2025-04-23 16:45:01,698 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (469.127893 ms). Ensure all imports are inside your registered functions.
    2025-04-23 16:45:02,024 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/intermediate/RAG/simple_rag/configs/milvus_rag_config.yml'
    2025-04-23 16:45:02,032 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
    2025-04-23 16:45:02,169 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
    2025-04-23 16:45:02,177 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.

    Configuration Summary:
    --------------------
    Workflow Type: react_agent
    Number of Functions: 2
    Number of LLMs: 1
    Number of Embedders: 1
    Number of Memory: 0
    Number of Retrievers: 2

    2025-04-23 16:45:03,203 - aiq.agent.react_agent.agent - INFO -
    ------------------------------
    [AGENT]
    Agent input: How do I install CUDA
    Agent's thoughts:
    Thought: To answer the user's question, I need to find information about installing CUDA.
    Action: cuda_retriever_tool
    Action Input: {"query": "install CUDA"}

    ------------------------------
    2025-04-23 16:45:03,511 - aiq.tool.retriever - INFO - Retrieved 10 records for query install CUDA.
    2025-04-23 16:45:03,513 - aiq.agent.react_agent.agent - INFO -
    ------------------------------
    [AGENT]
    Calling tools: cuda_retriever_tool
    Tool's input: {"query": "install CUDA"}
    Tool's response:
    {"results": [{"page_content": "1. Introduction \u2014 Installation Guide Windows 12.8 documentation\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n1. Introduction\n1.1. System Requirements\n1.2. About This Document\n\n\n2. Installing CUDA Development Tools\n2.1. Verify You Have a CUDA-capable GPU\n2.2. Download the NVIDIA CUDA Toolkit\n2.3. Install the CUDA Software\n2.3.1. Uninstalling the CUDA Software\n\n\n2.4. Using Conda to Install the CUDA Software\n2.4.1. Conda Overview\n2.4.2. Installation\n2.4.3. Uninstallation\n2.4.4. Installing Previous CUDA Releases\n\n\n2.5. Use a Suitable Driver Model\n2.6. Verify the Installation\n2.6.1. Running the Compiled Examples\n\n\n\n\n3. Pip Wheels\n4. Compiling CUDA Programs\n4.1. Compiling Sample Projects\n4.2. Sample Projects\n4.3. Build Customizations for New Projects\n4.4. Build Customizations for Existing Projects\n\n\n5. Additional Considerations\n6. Notices\n6.1. Notice\n6.2. OpenCL\n6.3. Trademarks\n\n\n...
    ------------------------------
    2025-04-23 16:45:06,407 - aiq.agent.react_agent.agent - INFO -
    ------------------------------
    [AGENT]
    Agent input: How do I install CUDA
    Agent's thoughts:
    Thought: The provided tool output contains detailed instructions for installing CUDA on various Linux distributions and Windows. To answer the user's question, I will summarize the general steps for installing CUDA.

    Final Answer: To install CUDA, you need to follow these general steps:

    1. Verify that your system has a CUDA-capable GPU.
    2. Choose an installation method: local repo or network repo.
    3. Download the NVIDIA CUDA Toolkit from the official NVIDIA website.
    4. Install the CUDA Toolkit using the chosen installation method.
    5. Perform post-installation actions, such as updating the Apt repository cache and installing additional packages.

    For specific instructions, please refer to the official NVIDIA documentation for your operating system: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html or https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html.
    ------------------------------
    2025-04-23 16:45:06,412 - aiq.front_ends.console.console_front_end_plugin - INFO -
    --------------------------------------------------
    Workflow Result:
    ['To install CUDA, you need to follow these general steps:\n\n1. Verify that your system has a CUDA-capable GPU.\n2. Choose an installation method: local repo or network repo.\n3. Download the NVIDIA CUDA Toolkit from the official NVIDIA website.\n4. Install the CUDA Toolkit using the chosen installation method.\n5. Perform post-installation actions, such as updating the Apt repository cache and installing additional packages.\n\nFor specific instructions, please refer to the official NVIDIA documentation for your operating system: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html or https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html.']
    --------------------------------------------------
    ```

## Adding Long-Term Agent Memory
If you want to add long-term memory to your agent, you can do so by adding a `memory` section to your configuration file. The memory section is used to store information that the agent can use to provide more contextually relevant answers to the user's questions. The memory section can be used to store information such as user preferences, past interactions, or any other information that the agent needs to remember.

### Prerequisites
This section requires an API key for integration with the Mem0 Platform. To create an API key, refer to the instructions in the [Mem0 Platform Guide](https://docs.mem0.ai/platform/quickstart). Once you have created your API key, export it as an environment variable:
```bash
export MEM0_API_KEY=<MEM0 API KEY HERE>
```

### Adding Memory to the Agent
Adding the ability to add and retrieve long-term memory to the agent is just a matter of adding a `memory` section to the configuration file. The AIQ toolkit built-in abstractions for long term memory management allow agents to automatically interact with them as tools. We will use the following configuration file, which you can also find in the `configs` directory.

```yaml
general:
  use_uvloop: true

memory:
  saas_memory:
    _type: mem0_memory

retrievers:
  cuda_retriever:
    _type: milvus_retriever
    uri: http://localhost:19530
    collection_name: "cuda_docs"
    embedding_model: milvus_embedder
    top_k: 10
  mcp_retriever:
    _type: milvus_retriever
    uri: http://localhost:19530
    collection_name: "mcp_docs"
    embedding_model: milvus_embedder
    top_k: 10

functions:
  cuda_retriever_tool:
    _type: aiq_retriever
    retriever: cuda_retriever
    topic: Retrieve documentation for NVIDIA's CUDA library
  mcp_retriever_tool:
    _type: aiq_retriever
    retriever: mcp_retriever
    topic: Retrieve information about Model Context Protocol (MCP)
  add_memory:
    _type: add_memory
    memory: saas_memory
    description: |
      Add any facts about user preferences to long term memory. Always use this if users mention a preference.
      The input to this tool should be a string that describes the user's preference, not the question or answer.
  get_memory:
    _type: get_memory
    memory: saas_memory
    description: |
      Always call this tool before calling any other tools, even if the user does not mention to use it.
      The question should be about user preferences which will help you format your response.
      For example: "How does the user like responses formatted?"

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.3-70b-instruct
    temperature: 0
    max_tokens: 4096
    top_p: 1

embedders:
  milvus_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5
    truncate: "END"

workflow:
  _type: react_agent
  tool_names:
   - cuda_retriever_tool
   - mcp_retriever_tool
   - add_memory
   - get_memory
  verbose: true
  llm_name: nim_llm
```

Notice in the configuration above that the only addition to the configuration that was required to add long term memory to the agent was a `memory` section in the configuration specifying:
- The type of memory to use (`mem0_memory`)
- The name of the memory (`saas_memory`)

Then, we used native AIQ toolkit functions for getting memory and adding memory to the agent. These functions are:
- `add_memory`: This function is used to add any facts about user preferences to long term memory.
- `get_memory`: This function is used to retrieve any facts about user preferences from long term memory.

Each function was given a description that helps the agent know when to use it as a tool. With the configuration in place, we can run the workflow again.
This time, we will tell the agent about how we like our responses formatted, and notice if it stores that fact to long term memory.

```bash
aiq run --config_file=examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml --input "How do I install CUDA? I like responses with a lot of emojis in them! :)"
```

The expected output of the above run is:

```console
$ aiq run --config_file=examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml --input "How do I install CUDA? I like responses with a lot of emojis in them! :)"
2025-04-23 16:56:40,025 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (478.030443 ms). Ensure all imports are inside your registered functions.
2025-04-23 16:56:40,222 - aiq.runtime.loader - WARNING - Loading module 'aiq_swe_bench.register' from entry point 'aiq_swe_bench' took a long time (103.739262 ms). Ensure all imports are inside your registered functions.
2025-04-23 16:56:40,376 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml'
2025-04-23 16:56:40,385 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 16:56:41,738 - httpx - INFO - HTTP Request: GET https://api.mem0.ai/v1/ping/ "HTTP/1.1 200 OK"
2025-04-23 16:56:41,933 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-04-23 16:56:41,946 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 1
Number of Retrievers: 2

2025-04-23 16:56:44,973 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA? I like responses with a lot of emojis in them! :)
Agent's thoughts:
Thought: The user is asking about installing CUDA, and they have a preference for responses with a lot of emojis. I should first try to retrieve information about the user's preference for response format.

Action: get_memory
Action Input: {"query": "response format", "top_k": 1, "user_id": "current_user"}

------------------------------
2025-04-23 16:56:45,143 - httpx - INFO - HTTP Request: POST https://api.mem0.ai/v1/memories/search/ "HTTP/1.1 200 OK"
2025-04-23 16:56:45,145 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: get_memory
Tool's input: {"query": "response format", "top_k": 1, "user_id": "current_user"}
Tool's response:
Memories as a JSON:
[{"conversation": [], "tags": ["user_preferences"], "metadata": {}, "user_id": "current_user", "memory": "Likes responses with a lot of emojis"}]
------------------------------
2025-04-23 16:56:45,875 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA? I like responses with a lot of emojis in them! :)
Agent's thoughts:
Thought: The user likes responses with a lot of emojis. Now, I should try to find information about installing CUDA.

Action: cuda_retriever_tool
Action Input: {"query": "install CUDA"}
------------------------------
2025-04-23 16:56:46,203 - aiq.tool.retriever - INFO - Retrieved 10 records for query install CUDA.
2025-04-23 16:56:46,230 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: cuda_retriever_tool
Tool's input: {"query": "install CUDA"}
Tool's response:
{"results": [{"page_content": "1. Introduction \u2014 Installation Guide Windows 12.8 documentation\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n1. Introduction\n1.1. System Requirements\n1.2. About This Document\n\n\n2. Installing CUDA Development Tools\n2.1. Verify You Have a CUDA-capable GPU\n2.2. Download the NVIDIA CUDA Toolkit\n2.3. Install the CUDA Software\n2.3.1. Uninstalling the CUDA Software\n\n\n2.4. Using Conda to Install the CUDA Software\n2.4.1. Conda Overview\n2.4.2. Installation\n2.4.3. Uninstallation\n2.4.4. Installing Previous CUDA Releases\n\n\n2.5. Use a Suitable Driver Model\n2.6. Verify the Installation\n2.6.1. Running the Compiled Examples\n\n\n\n\n3. Pip Wheels\n4. Compiling CUDA Programs\n4.1. Compiling Sample Projects\n4.2. Sample Projects\n4.3. Build Customizations for New Projects\n4.4. Build Customizations for Existing Projects\n\n\n5. Additional Considerations\n6. Notices\n6.1. Notice\n6.2. OpenCL\n6.3. Trademarks\n\n\n...
------------------------------
2025-04-23 16:56:51,636 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA? I like responses with a lot of emojis in them! :)
Agent's thoughts:
Thought: The user likes responses with a lot of emojis. Now, I should try to find information about installing CUDA.

The installation instructions for CUDA are as follows:

1. Verify you have a CUDA-capable GPU.
2. Download the NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads.
3. Install the NVIDIA CUDA Toolkit.

For Windows:
- Graphical Installation: Execute the CUDA installer and follow the on-screen prompts.
- Silent Installation: Execute the package with the -s flag.

For Linux:
- Package Manager Installation: Install using RPM or Debian packages, which interface with your system's package management system.
- Runfile Installation: Install using the standalone installer, a .run file that is completely self-contained.

After installation, perform the post-installation actions.

Thought: I now know the final answer

Final Answer: 🎉👍 To install CUDA, follow these steps: 📝
1. Verify you have a CUDA-capable GPU 🖥️.
2. Download the NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads 📊.
3. Install the NVIDIA CUDA Toolkit 📈.
For Windows:
- Graphical Installation: Execute the CUDA installer and follow the on-screen prompts 📺.
- Silent Installation: Execute the package with the -s flag 🗣️.
For Linux:
- Package Manager Installation: Install using RPM or Debian packages, which interface with your system’s package management system 📦.
- Runfile Installation: Install using the standalone installer, a .run file that is completely self-contained 📁.
After installation, perform the post-installation actions 📝. 🎉👍
------------------------------
2025-04-23 16:56:51,642 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['🎉👍 To install CUDA, follow these steps: 📝\n1. Verify you have a CUDA-capable GPU 🖥️.\n2. Download the NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads 📊.\n3. Install the NVIDIA CUDA Toolkit 📈.\nFor Windows: \n- Graphical Installation: Execute the CUDA installer and follow the on-screen prompts 📺.\n- Silent Installation: Execute the package with the -s flag 🗣️.\nFor Linux: \n- Package Manager Installation: Install using RPM or Debian packages, which interface with your system’s package management system 📦.\n- Runfile Installation: Install using the standalone installer, a .run file that is completely self-contained 📁.\nAfter installation, perform the post-installation actions 📝. 🎉👍']
--------------------------------------------------
```

Notice above that the agent called the `add_memory` tool after retrieving the information about installing CUDA. The `add_memory` tool was given the conversation between the user and the assistant, the tags for the memory, and the metadata for the memory.

Now, we can try another invocation of the agent without mentioning our preference to see if it remembers our preference from the previous conversation.

```bash
aiq run --config_file=examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml --input "How do I install CUDA?"
```

The expected output of the above run is:

```console
$ aiq run --config_file=examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml --input "How do I install CUDA?"
2025-04-23 16:59:21,197 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (444.273233 ms). Ensure all imports are inside your registered functions.
2025-04-23 16:59:21,517 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/intermediate/RAG/simple_rag/configs/milvus_memory_rag_config.yml'
2025-04-23 16:59:21,525 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 16:59:22,833 - httpx - INFO - HTTP Request: GET https://api.mem0.ai/v1/ping/ "HTTP/1.1 200 OK"
2025-04-23 16:59:23,029 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-04-23 16:59:23,041 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 1
Number of Retrievers: 2

2025-04-23 16:59:24,882 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA?
Agent's thoughts:
Thought: To answer the user's question, I need to find information about installing CUDA. I should use the get_memory tool to see if the user has any preferences for how the response should be formatted.

Action: get_memory
Action Input: {"query": "response format preferences", "top_k": 1, "user_id": "current_user"}

------------------------------
2025-04-23 16:59:25,049 - httpx - INFO - HTTP Request: POST https://api.mem0.ai/v1/memories/search/ "HTTP/1.1 200 OK"
2025-04-23 16:59:25,051 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: get_memory
Tool's input: {"query": "response format preferences", "top_k": 1, "user_id": "current_user"}
Tool's response:
Memories as a JSON:
[{"conversation": [], "tags": ["user_preferences"], "metadata": {}, "user_id": "current_user", "memory": "Likes responses with a lot of emojis"}]
------------------------------
2025-04-23 16:59:27,888 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA?
Agent's thoughts:
Thought: The user likes responses with a lot of emojis. Now, I need to find information about installing CUDA. I should use the cuda_retriever_tool to find the installation steps.

Action: cuda_retriever_tool
Action Input: {"query": "install CUDA"}
------------------------------
2025-04-23 16:59:28,217 - aiq.tool.retriever - INFO - Retrieved 10 records for query install CUDA.
2025-04-23 16:59:28,221 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: cuda_retriever_tool
Tool's input: {"query": "install CUDA"}
Tool's response:
{"results": [{"page_content": "1. Introduction \u2014 Installation Guide Windows 12.8 documentation\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n1. Introduction\n1.1. System Requirements\n1.2. About This Document\n\n\n2. Installing CUDA Development Tools\n2.1. Verify You Have a CUDA-capable GPU\n2.2. Download the NVIDIA CUDA Toolkit\n2.3. Install the CUDA Software\n2.3.1. Uninstalling the CUDA Software\n\n\n2.4. Using Conda to Install the CUDA Software\n2.4.1. Conda Overview\n2.4.2. Installation\n2.4.3. Uninstallation\n2.4.4. Installing Previous CUDA Releases\n\n\n2.5. Use a Suitable Driver Model\n2.6. Verify the Installation\n2.6.1. Running the Compiled Examples\n\n\n\n\n3. Pip Wheels\n4. Compiling CUDA Programs\n4.1. Compiling Sample Projects\n4.2. Sample Projects\n4.3. Build Customizations for New Projects\n4.4. Build Customizations for Existing Projects\n\n\n5. Additional Considerations\n6. Notices\n6.1. Notice\n6.2. OpenCL\n6.3. Trademarks\n\n\n...
------------------------------
2025-04-23 16:59:36,163 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA?
Agent's thoughts:
Thought: The user likes responses with a lot of emojis. Now, I need to find information about installing CUDA. The results from the cuda_retriever_tool provide detailed steps for installing CUDA on various operating systems, including Windows and Linux.

To install CUDA on Windows, the user should:
1. Verify that their system has a CUDA-capable GPU 🖥️
2. Download the NVIDIA CUDA Toolkit from the official website 📦
3. Install the CUDA Software by executing the CUDA installer and following the on-screen prompts 💻
4. Test that the installed software runs correctly and communicates with the hardware 📊

To install CUDA on Linux, the user should:
1. Perform the pre-installation actions 📝
2. Choose an installation method: local repo or network repo 🤔
3. Install the CUDA Toolkit using the chosen method 📦
4. Perform the post-installation actions 📈

Here are the detailed steps for each operating system:

**Windows:**
1. Verify that your system has a CUDA-capable GPU 🖥️
2. Download the NVIDIA CUDA Toolkit from the official website 📦
3. Install the CUDA Software by executing the CUDA installer and following the on-screen prompts 💻
4. Test that the installed software runs correctly and communicates with the hardware 📊

**Linux:**
1. Perform the pre-installation actions 📝
2. Choose an installation method: local repo or network repo 🤔
3. Install the CUDA Toolkit using the chosen method 📦
4. Perform the post-installation actions 📈

For example, on Ubuntu, the user can install CUDA using the following commands:

  sudo dpkg -i cuda-repo-<distro>-X-Y-local_<version>*_<arch>.deb
  sudo cp /var/cuda-repo-<distro>-X-Y-local/cuda-*-keyring.gpg /usr/share/keyrings/
  sudo mv cuda-<distro>.pin /etc/apt/preferences.d/cuda-repository-pin-600
  sudo apt-get update
  sudo apt-get install cuda-toolkit

Replace `<distro>`, `<version>`, and `<arch>` with the corresponding values for your system 🤖

Final Answer: 🎉 To install CUDA, follow these steps:
1. Verify that your system has a CUDA-capable GPU 🖥️
2. Download the NVIDIA CUDA Toolkit from the official website 📦
3. Install the CUDA Software by executing the CUDA installer and following the on-screen prompts 💻
4. Test that the installed software runs correctly and communicates with the hardware 📊
For Linux, choose an installation method and install the CUDA Toolkit using the chosen method 📦
Remember to perform the post-installation actions 📈
👍
------------------------------
2025-04-23 16:59:36,167 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['🎉 To install CUDA, follow these steps: \n1. Verify that your system has a CUDA-capable GPU 🖥️\n2. Download the NVIDIA CUDA Toolkit from the official website 📦\n3. Install the CUDA Software by executing the CUDA installer and following the on-screen prompts 💻\n4. Test that the installed software runs correctly and communicates with the hardware 📊\nFor Linux, choose an installation method and install the CUDA Toolkit using the chosen method 📦\nRemember to perform the post-installation actions 📈\n👍']
--------------------------------------------------
```

We see from the above output that the agent was able to successfully retrieve our preference for emoji's in responses from long term memory and use it to format the response to our question about installing CUDA.

In this way, you can easily construct an agent that answers questions about your knowledge base and stores long term memories, all without any agent code required!

Note: The long-term memory feature relies on LLM-based tool invocation, which can occasionally be non-deterministic. If you notice that the memory functionality isn't working as expected (e.g., the agent doesn't remember your preferences), simply re-run your first and second inputs. This will help ensure the memory tools are properly invoked and your preferences are correctly stored.

## Adding Additional Tools
This workflow can be further enhanced by adding additional tools. Included with this example are two additional tools: `tavily_internet_search` and `code_generation`.

Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com`](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```
or update the workflow config file to include the `api_key`.

These workflows demonstrate how agents can use multiple tools in tandem to provide more robust responses. Both `milvus_memory_rag_tools_config.yml` and `milvus_rag_tools_config.yml` use these additional tools.

We can now run one of these workflows with a slightly more complex input.

```bash
aiq run --config_file examples/intermediate/RAG/simple_rag/configs/milvus_rag_tools_config.yml --input "How do I install CUDA and get started developing with it? Provide example python code"
```
The expected output of the above run is:
````console
$ aiq run --config_file examples/intermediate/RAG/simple_rag/configs/milvus_rag_tools_config.yml --input "How do I install CUDA and get started developing with it? Provide example python code"
2025-04-23 20:31:34,456 - aiq.runtime.loader - WARNING - Loading module 'aiq_automated_description_generation.register' from entry point 'aiq_automated_description_generation' took a long time (491.573811 ms). Ensure all imports are inside your registered functions.
2025-04-23 20:31:34,779 - aiq.cli.commands.start - INFO - Starting AIQ toolkit from config file: 'examples/intermediate/RAG/simple_rag/configs/milvus_rag_tools_config.yml'
2025-04-23 20:31:34,788 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-04-23 20:31:34,950 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-04-23 20:31:34,960 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-04-23 20:31:34,964 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function code_generation_tool by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-04-23 20:31:34,966 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initializing code generation tool
Getting tool LLM from config
2025-04-23 20:31:34,968 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Filling tool's prompt variable from config
2025-04-23 20:31:34,968 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initialized code generation tool

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 2

2025-04-23 20:31:36,778 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA and get started developing with it? Provide example python code
Agent's thoughts:
Thought: To answer this question, I need to provide information on how to install CUDA and get started with developing applications using it. I also need to provide example Python code to demonstrate its usage.

Action: cuda_retriever_tool
Action Input: {"query": "installing CUDA and getting started"}

------------------------------
2025-04-23 20:31:37,097 - aiq.tool.retriever - INFO - Retrieved 10 records for query installing CUDA and getting started.
2025-04-23 20:31:37,099 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: cuda_retriever_tool
Tool's input: {"query": "installing CUDA and getting started"}
Tool's response:
{"results": [{"page_content": "1. Introduction \u2014 Installation Guide Windows 12.8 documentation\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n1. Introduction\n1.1. System Requirements\n1.2. About This Document\n\n\n2. Installing CUDA Development Tools\n2.1. Verify You Have a CUDA-capable GPU\n2.2. Download the NVIDIA CUDA Toolkit\n2.3. Install the CUDA Software\n2.3.1. Uninstalling the CUDA Software\n\n\n2.4. Using Conda to Install the CUDA Software\n2.4.1. Conda Overview\n2.4.2. Installation\n2.4.3. Uninstallation\n2.4.4. Installing Previous CUDA Releases\n\n\n2.5. Use a Suitable Driver Model\n2.6. Verify the Installation\n2.6.1. Running the Compiled Examples\n\n\n\n\n3. Pip Wheels\n4. Compiling CUDA Programs\n4.1. Compiling Sample Projects\n4.2. Sample Projects\n4.3. Build Customizations for New Projects\n4.4. Build Customizations for Existing Projects\n\n\n5. Additional Considerations\n6. Notices\n6.1. Notice\n6.2. OpenCL\n6.3. Trademarks\n\n\n...
------------------------------
2025-04-23 20:31:46,243 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA and get started developing with it? Provide example python code
Agent's thoughts:
Thought: The provided information from the CUDA documentation gives a detailed guide on how to install CUDA on various operating systems, including Windows and Linux. It also provides information on how to verify the installation and run sample programs to test the setup. However, to provide example Python code, I need to use the code generation tool.

Action: code_generation_tool
Action Input: {"query": "Python code example using CUDA"}
------------------------------
2025-04-23 20:31:46,251 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Running code generation tool
2025-04-23 20:31:47,931 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: code_generation_tool
Tool's input: {"query": "Python code example using CUDA"}
Tool's response:
```python
import numpy as np
import cupy as cp

# Create a sample array
arr = np.array([1, 2, 3, 4, 5])

# Transfer the array to the GPU
arr_gpu = cp.asarray(arr)

# Perform some operations on the GPU
result_gpu = cp.square(arr_gpu)

# Transfer the result back to the CPU
result_cpu = cp.asnumpy(result_gpu)

print(result_cpu)
```
------------------------------
2025-04-23 20:31:52,241 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: How do I install CUDA and get started developing with it? Provide example python code
Agent's thoughts:
Thought: I now know the final answer

Final Answer: To install CUDA and get started with developing applications using it, you can follow the steps outlined in the CUDA documentation. This includes verifying that you have a CUDA-capable GPU, downloading the NVIDIA CUDA Toolkit, and installing the CUDA software. After installation, you can verify that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware by compiling and running sample programs.

Here's an example Python code that demonstrates how to use CUDA:
```python
import numpy as np
import cupy as cp

# Create a sample array
arr = np.array([1, 2, 3, 4, 5])

# Transfer the array to the GPU
arr_gpu = cp.asarray(arr)

# Perform some operations on the GPU
result_gpu = cp.square(arr_gpu)

# Transfer the result back to the CPU
result_cpu = cp.asnumpy(result_gpu)

print(result_cpu)
```
This code creates a sample array, transfers it to the GPU, performs some operations on the GPU, and then transfers the result back to the CPU. The output of this code will be the squared values of the original array.
------------------------------
2025-04-23 20:31:52,244 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
["To install CUDA and get started with developing applications using it, you can follow the steps outlined in the CUDA documentation. This includes verifying that you have a CUDA-capable GPU, downloading the NVIDIA CUDA Toolkit, and installing the CUDA software. After installation, you can verify that the CUDA toolkit can find and communicate correctly with the CUDA-capable hardware by compiling and running sample programs.\n\nHere's an example Python code that demonstrates how to use CUDA:\n```python\nimport numpy as np\nimport cupy as cp\n\n# Create a sample array\narr = np.array([1, 2, 3, 4, 5])\n\n# Transfer the array to the GPU\narr_gpu = cp.asarray(arr)\n\n# Perform some operations on the GPU\nresult_gpu = cp.square(arr_gpu)\n\n# Transfer the result back to the CPU\nresult_cpu = cp.asnumpy(result_gpu)\n\nprint(result_cpu)\n```\nThis code creates a sample array, transfers it to the GPU, performs some operations on the GPU, and then transfers the result back to the CPU. The output of this code will be the squared values of the original array."]
--------------------------------------------------
````
