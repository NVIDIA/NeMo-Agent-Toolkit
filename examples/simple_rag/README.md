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

1) Start the docker compose [Skip this step if you already have Milvus running]
    ```bash
    docker compose -f examples/simple_rag/deploy/docker-compose.yaml up -d
    ```
2) In a new terminal, from the root of the AgentIQ repository, run the provided bash script to store the data in a Milvus collection. By default the script will scrape a few pages from the CUDA documentation and store the data in a Milvus collection called `cuda_docs`. It will also pull a few pages of information about the Model Context Protocol (MCP) and store it in a collection called `mcp_docs`.

    Export your NVIDIA API key:
    ```bash
    export NVIDIA_API_KEY=<YOUR API KEY HERE>
    ```

    ```bash
    source .venv/bin/activate
    examples/simple_rag/ingestion/bootstrap_milvus.sh
    ```

    If Milvus is running the script should work out of the box. If you want to customize the script the arguments are shown below.
    ```bash
    python examples/simple_rag/ingestion/langchain_web_ingest.py --help
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

3) Configure your Agent to use the Milvus collections for RAG. We have pre-configured a configuration file for you in `examples/simple_rag/configs/milvus_rag_config.yml`. You can modify this file to point to your Milvus instance and collections or add tools to your agent. The agent, by default, is a `tool_calling` agent that can be used to interact with the retriever component. The configuration file is shown below. You can also modify your agent to be another one of AgentIQ's pre-built agent implementations
such as the `react_agent`

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

4) Run the workflow
    ```bash
    aiq run --config_file examples/simple_rag/configs/milvus_rag_config.yml --input "How do I install CUDA"
    ```
   The expected output of running the above command is:
    ```console
   2025-03-11 15:19:18,551 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/simple_rag/configs/milvus_rag_config.yml'
    2025-03-11 15:19:18,556 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
    2025-03-11 15:19:19,634 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
    2025-03-11 15:19:19,831 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
    2025-03-11 15:19:19,833 - aiq.profiler.decorators - INFO - Langchain callback handler registered
    2025-03-11 15:19:20,237 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
    2025-03-11 15:19:20,238 - aiq.agent.react_agent.agent - INFO - Adding the tools' input schema to the tools' description
    2025-03-11 15:19:20,238 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
    2025-03-11 15:19:20,242 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully
    2025-03-11 15:19:20,243 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('How do I install CUDA?',)
    2025-03-11 15:19:20,246 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1

    Configuration Summary:
    --------------------
    Workflow Type: react_agent
    Number of Functions: 2
    Number of LLMs: 1
    Number of Embedders: 1
    Number of Memory: 0
    Number of Retrievers: 2

    2025-03-11 15:19:21,142 - aiq.agent.react_agent.agent - INFO - The user's question was: How do I install CUDA?
    2025-03-11 15:19:21,142 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
    Thought: To answer the user's question, I need to find information about installing CUDA.
    Action: cuda_retriever_tool
    Action Input: {"query": "install CUDA"}

    2025-03-11 15:19:21,146 - aiq.agent.react_agent.agent - INFO - Calling tool cuda_retriever_tool with input: {"query": "install CUDA"}
    2025-03-11 15:19:21,146 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
    2025-03-11 15:19:21,853 - aiq.tool.retriever - INFO - Retrieved 10 records for query install CUDA.
    2025-03-11 15:19:21,855 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
    2025-03-11 15:19:25,078 - aiq.agent.react_agent.agent - INFO -

    The agent's thoughts are:
    Thought: The provided tool output contains detailed instructions for installing CUDA on various Linux distributions and Windows. To provide a clear and concise answer, I will summarize the general steps for installing CUDA.

    Final Answer: To install CUDA, you need to follow these general steps:

    1. Verify that your system has a CUDA-capable GPU.
       2. Choose an installation method: network installer, local installer, or package manager installation.
       3. Download the NVIDIA CUDA Toolkit from the official website.
       4. Install the CUDA Toolkit using the chosen installation method.
       5. Perform post-installation actions, such as updating the Apt repository cache and installing additional packages.

    For specific instructions, please refer to the official NVIDIA CUDA documentation, which provides detailed guides for various Linux distributions and Windows.
    2025-03-11 15:19:25,083 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
    2025-03-11 15:19:25,084 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
    Workflow Result:
    ['To install CUDA, you need to follow these general steps:\n\n1. Verify that your system has a CUDA-capable GPU.\n2. Choose an installation method: network installer, local installer, or package manager installation.\n3. Download the NVIDIA CUDA Toolkit from the official website.\n4. Install the CUDA Toolkit using the chosen installation method.\n5. Perform post-installation actions, such as updating the Apt repository cache and installing additional packages.\n\nFor specific instructions, please refer to the official NVIDIA CUDA documentation, which provides detailed guides for various Linux distributions and Windows.']
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
Adding the ability to add and retrieve long-term memory to the agent is just a matter of adding a `memory` section to the configuration file. The `memory` section should contain the following fields:
AgentIQ's native abstractions for long term memory management allow agents to automatically interact with them as tools. We will use the following configuration file, which you can also find in the `configs` directory.

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

Then, we used native AgentIQ functions for getting memory and adding memory to the agent. These functions are:
- `add_memory`: This function is used to add any facts about user preferences to long term memory.
- `get_memory`: This function is used to retrieve any facts about user preferences from long term memory.

Each function was given a description that helps the agent know when to use it as a tool. With the configuration in place, we can run the workflow again.
This time, we will tell the agent about how we like our responses formatted, and notice if it stores that fact to long term memory.

```bash
aiq run --config_file=examples/simple_rag/configs/milvus_memory_rag_config.yml --input "How do I install CUDA? I like responses with a lot of emojis in them! :)"
```

The expected output of the above run is:

```console
2025-03-11 15:42:18,804 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/simple_rag/configs/milvus_memory_rag_config.yml'
2025-03-11 15:42:18,807 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-03-11 15:42:20,301 - httpx - INFO - HTTP Request: GET https://api.mem0.ai/v1/ping/ "HTTP/1.1 200 OK"
2025-03-11 15:42:21,073 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-03-11 15:42:21,283 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-03-11 15:42:21,295 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-11 15:42:21,692 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-03-11 15:42:21,692 - aiq.agent.react_agent.agent - INFO - Adding the tools' input schema to the tools' description
2025-03-11 15:42:21,692 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-03-11 15:42:21,696 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully
2025-03-11 15:42:21,697 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('How do I install CUDA? I like responses with a lot of emojis in them! :)',)
2025-03-11 15:42:21,699 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 1
Number of Retrievers: 2

2025-03-11 15:42:22,935 - aiq.agent.react_agent.agent - INFO - The user's question was: How do I install CUDA? I like responses with a lot of emojis in them! :)
2025-03-11 15:42:22,936 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: The user is asking for installation instructions for CUDA and has a preference for responses with a lot of emojis. I should first retrieve information about installing CUDA and then add the user's preference to long term memory.

Action: cuda_retriever_tool
Action Input: {"query": "install CUDA"}

2025-03-11 15:42:22,942 - aiq.agent.react_agent.agent - INFO - Calling tool cuda_retriever_tool with input: {"query": "install CUDA"}
2025-03-11 15:42:22,942 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-11 15:42:23,584 - aiq.tool.retriever - INFO - Retrieved 10 records for query install CUDA.
2025-03-11 15:42:23,588 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-11 15:42:26,993 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I have retrieved the information about installing CUDA from the NVIDIA documentation. Now, I need to add the user's preference for responses with a lot of emojis to long term memory.

Action: add_memory
Action Input: {"conversation": [{"role": "user", "content": "How do I install CUDA? I like responses with a lot of emojis! :)"}, {"role": "assistant", "content": "To install CUDA, you can follow these steps: ..."}], "tags": ["CUDA", "installation"], "metadata": {"user_id": "1234", "context": "CUDA installation"}, "user_id": "1234", "memory": "None"}


2025-03-11 15:42:26,997 - aiq.agent.react_agent.agent - INFO - Calling tool add_memory with input: {"conversation": [{"role": "user", "content": "How do I install CUDA? I like responses with a lot of emojis! :)"}, {"role": "assistant", "content": "To install CUDA, you can follow these steps: ..."}], "tags": ["CUDA", "installation"], "metadata": {"user_id": "1234", "context": "CUDA installation"}, "user_id": "1234", "memory": "None"}

2025-03-11 15:42:26,997 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-11 15:42:29,163 - httpx - INFO - HTTP Request: POST https://api.mem0.ai/v1/memories/ "HTTP/1.1 200 OK"
2025-03-11 15:42:29,167 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-11 15:42:32,413 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I have added the user's preference to long term memory. Now, I can respond to the user's question about installing CUDA with a lot of emojis.

Final Answer: 🎉👍 To install CUDA, you can follow these steps: 📝

1. Verify you have a CUDA-capable GPU 🖥️
2. Download the NVIDIA CUDA Toolkit 📦
3. Install the CUDA Software 📈
4. Test that the installed software runs correctly and communicates with the hardware 📊

👍 You can find more detailed instructions in the NVIDIA CUDA documentation 📚. 🤔 If you have any questions or need further assistance, feel free to ask 🤗! 😊
2025-03-11 15:42:32,416 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-11 15:42:32,416 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['🎉👍 To install CUDA, you can follow these steps: 📝\n\n1. Verify you have a CUDA-capable GPU 🖥️\n2. Download the NVIDIA CUDA Toolkit 📦\n3. Install the CUDA Software 📈\n4. Test that the installed software runs correctly and communicates with the hardware 📊\n\n👍 You can find more detailed instructions in the NVIDIA CUDA documentation 📚. 🤔 If you have any questions or need further assistance, feel free to ask 🤗! 😊']
--------------------------------------------------
```

Notice above that the agent called the `add_memory` tool after retrieving the information about installing CUDA. The `add_memory` tool was given the conversation between the user and the assistant, the tags for the memory, and the metadata for the memory.

Now, we can try another invocation of the agent without mentioning our preference to see if it remembers our preference from the previous conversation.

```bash
aiq run --config_file=examples/simple_rag/configs/milvus_memory_rag_config.yml --input "How do I install CUDA?"
```

The expected output of the above run is:

```console
2025-03-11 15:54:23,700 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/simple_rag/configs/milvus_memory_rag_config.yml'
2025-03-11 15:54:23,704 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-03-11 15:54:24,831 - httpx - INFO - HTTP Request: GET https://api.mem0.ai/v1/ping/ "HTTP/1.1 200 OK"
2025-03-11 15:54:25,632 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-03-11 15:54:25,856 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-03-11 15:54:25,867 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-11 15:54:26,251 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-03-11 15:54:26,251 - aiq.agent.react_agent.agent - INFO - Adding the tools' input schema to the tools' description
2025-03-11 15:54:26,251 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-03-11 15:54:26,255 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully
2025-03-11 15:54:26,256 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('How do I install CUDA?',)
2025-03-11 15:54:26,258 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 1
Number of Retrievers: 2

2025-03-11 15:54:27,518 - aiq.agent.react_agent.agent - INFO - The user's question was: How do I install CUDA?
2025-03-11 15:54:27,519 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To answer the user's question, I need to find information about installing CUDA. I should first check if the user has any preferences for the response format.

Action: get_memory
Action Input: {"query": "response format preference", "top_k": 1, "user_id": "current_user"}

2025-03-11 15:54:27,525 - aiq.agent.react_agent.agent - INFO - Calling tool get_memory with input: {"query": "response format preference", "top_k": 1, "user_id": "current_user"}
2025-03-11 15:54:27,525 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-11 15:54:27,884 - httpx - INFO - HTTP Request: POST https://api.mem0.ai/v1/memories/search/ "HTTP/1.1 200 OK"
2025-03-11 15:54:27,888 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-11 15:54:28,865 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The user loves a lot of emojis in responses. Now, I need to find information about installing CUDA.

Action: cuda_retriever_tool
Action Input: {"query": "installing CUDA"}
2025-03-11 15:54:28,869 - aiq.agent.react_agent.agent - INFO - Calling tool cuda_retriever_tool with input: {"query": "installing CUDA"}
2025-03-11 15:54:28,869 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-11 15:54:29,541 - aiq.tool.retriever - INFO - Retrieved 10 records for query installing CUDA.
2025-03-11 15:54:29,549 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-11 15:54:36,376 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The user loves a lot of emojis in responses. Now, I have found information about installing CUDA.

To install CUDA, you need to follow these steps:
1. Verify you have a CUDA-capable GPU.
2. Download the NVIDIA CUDA Toolkit from https://developer.nvidia.com/cuda-downloads.
3. Install the CUDA Software by executing the CUDA installer and following the on-screen prompts.
4. Test that the installed software runs correctly and communicates with the hardware.

Here are the steps with more details and emojis:
🌟 Step 1: Verify you have a CUDA-capable GPU 🌟
You can verify that you have a CUDA-capable GPU through the Display Adapters section in the Windows Device Manager 📊.

🌟 Step 2: Download the NVIDIA CUDA Toolkit 🌟
The NVIDIA CUDA Toolkit is available at https://developer.nvidia.com/cuda-downloads 🌐. Choose the platform you are using and one of the following installer formats: Network Installer or Full Installer 📦.

🌟 Step 3: Install the CUDA Software 🌟
Before installing the toolkit, you should read the Release Notes, as they provide details on installation and software functionality 📄. The setup of CUDA development tools on a system running the appropriate version of Windows consists of a few simple steps:
- Verify the system has a CUDA-capable GPU 🌟.
- Download the NVIDIA CUDA Toolkit 📦.
- Install the NVIDIA CUDA Toolkit 📈.
- Test that the installed software runs correctly and communicates with the hardware 📊.

🌟 Step 4: Test the Installation 🌟
Perform the post-installation actions 📝.

Final Answer: 🎉 To install CUDA, follow the steps: verify you have a CUDA-capable GPU 🌟, download the NVIDIA CUDA Toolkit 📦, install the CUDA Software 📈, and test the installation 📊. 🎊
2025-03-11 15:54:36,381 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-11 15:54:36,381 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['🎉 To install CUDA, follow the steps: verify you have a CUDA-capable GPU 🌟, download the NVIDIA CUDA Toolkit 📦, install the CUDA Software 📈, and test the installation 📊. 🎊']
--------------------------------------------------
```

We see from the above output that the agent was able to successfully retrieve our preference for emoji's in responses from long term memory and use it to format the response to our question about installing CUDA.

In this way, you can easily construct an agent that answers questions about your knowledge base and stores long term memories, all without any agent code required!

## Adding Additional Tools
This workflow can be further enhanced by adding additional tools. Included with this example are two additional tools: `tavily_internet_search` and `code_generation`. Both of these tools require the installation of the `agentiq[langchain]` package. To install this package run:
```bash
uv pip install -e '.[langchain]'
```
Prior to using the `tavily_internet_search` tool, create an account at [`tavily.com``](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```
or update the workflow config file to include the `api_key`.

These workflows demonstrate how agents can use multiple tools in tandem to provide more robust responses. Both `milvus_memory_rag_tools_config.yml` and `milvus_rag_tools_config.yml` use these additional tools.

We can now run one of these workflows with a slightly more complex input.

```bash
aiq run --config_file examples/simple_rag/configs/milvus_rag_tools_config.yml --input "How do I install CUDA and get started developing with it? Provide example python code"
```
The expected output of the above run is:
```console
2025-03-12 12:32:58,855 - aiq.runtime.loader - WARNING - Loading module 'aiq_simple.register' from entry point 'aiq_simple' took a long time (101.465940 ms). Ensure all imports are inside your registered functions.
2025-03-12 12:32:59,051 - aiq.cli.commands.start - INFO - Starting AgentIQ from config file: 'examples/simple_rag/configs/milvus_rag_tools_config.yml'
2025-03-12 12:32:59,057 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-03-12 12:32:59,764 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-03-12 12:32:59,769 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-03-12 12:32:59,771 - aiq.profiler.utils - WARNING - Discovered frameworks: {<LLMFrameworkEnum.LANGCHAIN: 'langchain'>} in function code_generation_tool by inspecting source. It is recommended and more reliable to instead add the used LLMFrameworkEnum types in the framework_wrappers argument when calling @register_function.
2025-03-12 12:32:59,773 - aiq.profiler.decorators - INFO - Langchain callback handler registered
2025-03-12 12:32:59,773 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initializing code generation tool
Getting tool LLM from config
2025-03-12 12:32:59,778 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Filling tool's prompt variable from config
2025-03-12 12:32:59,778 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Initialized code generation tool
2025-03-12 12:33:00,103 - aiq.agent.react_agent.agent - INFO - Filling the prompt variables "tools" and "tool_names", using the tools provided in the config.
2025-03-12 12:33:00,103 - aiq.agent.react_agent.agent - INFO - Adding the tools' input schema to the tools' description
2025-03-12 12:33:00,103 - aiq.agent.react_agent.agent - INFO - Initialized ReAct Agent Graph
2025-03-12 12:33:00,108 - aiq.agent.react_agent.agent - INFO - ReAct Graph built and compiled successfully

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 2

2025-03-12 12:33:00,109 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('How do I install CUDA and get started developing with it? Provide example python code',)
2025-03-12 12:33:00,113 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 12:33:01,462 - aiq.agent.react_agent.agent - INFO - The user's question was: How do I install CUDA and get started developing with it? Provide example python code
2025-03-12 12:33:01,462 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To answer this question, I need to provide information on how to install CUDA and get started with developing applications using it. I also need to provide example Python code to demonstrate its usage.

Action: cuda_retriever_tool
Action Input: {"query": "install CUDA and get started"}

2025-03-12 12:33:01,464 - aiq.agent.react_agent.agent - INFO - Calling tool cuda_retriever_tool with input: {"query": "install CUDA and get started"}
2025-03-12 12:33:01,464 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-12 12:33:01,985 - aiq.tool.retriever - INFO - Retrieved 10 records for query install CUDA and get started.
2025-03-12 12:33:01,988 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 12:33:03,896 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The provided information includes instructions on how to install CUDA on various operating systems, including Windows and Linux. It also mentions the system requirements, such as having a CUDA-capable GPU and a supported version of the operating system.

Action: code_generation_tool
Action Input: {"query": "example python code using CUDA"}
2025-03-12 12:33:03,898 - aiq.agent.react_agent.agent - INFO - Calling tool code_generation_tool with input: {"query": "example python code using CUDA"}
2025-03-12 12:33:03,898 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-12 12:33:03,900 - aiq.plugins.langchain.tools.code_generation_tool - INFO - Running code generation tool
2025-03-12 12:33:05,445 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-12 12:33:10,982 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: The provided code is an example of using CUDA with Python. It creates a sample array, moves it to the GPU, performs some operations on the GPU, and then moves the result back to the CPU.

Final Answer: To install CUDA and get started with developing applications using it, you can follow these steps:

1. Verify that your system has a CUDA-capable GPU.
2. Download the NVIDIA CUDA Toolkit from the official NVIDIA website.
3. Install the CUDA Toolkit using the installation guide provided for your operating system.
4. Test the installation by running the provided examples or by using the CUDA API in your own applications.

Here's an example Python code using CUDA:
```python
import numpy as np
import cupy as cp

# Create a sample array
arr = np.array([1, 2, 3, 4, 5])

# Move the array to the GPU
arr_gpu = cp.asarray(arr)

# Perform some operations on the GPU
result_gpu = cp.square(arr_gpu)

# Move the result back to the CPU
result_cpu = cp.asnumpy(result_gpu)

print(result_cpu)

This code creates a sample array, moves it to the GPU, performs a square operation on the GPU, and then moves the result back to the CPU. The result is then printed to the console.
2025-03-12 12:33:10,983 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-12 12:33:10,983 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
["To install CUDA and get started with developing applications using it, you can follow these steps:\n\n1. Verify that your system has a CUDA-capable GPU.\n2. Download the NVIDIA CUDA Toolkit from the official NVIDIA website.\n3. Install the CUDA Toolkit using the installation guide provided for your operating system.\n4. Test the installation by running the provided examples or by using the CUDA API in your own applications.\n\nHere's an example Python code using CUDA:\n```python\nimport numpy as np\nimport cupy as cp\n\n# Create a sample array\narr = np.array([1, 2, 3, 4, 5])\n\n# Move the array to the GPU\narr_gpu = cp.asarray(arr)\n\n# Perform some operations on the GPU\nresult_gpu = cp.square(arr_gpu)\n\n# Move the result back to the CPU\nresult_cpu = cp.asnumpy(result_gpu)\n\nprint(result_cpu)\n```\nThis code creates a sample array, moves it to the GPU, performs a square operation on the GPU, and then moves the result back to the CPU. The result is then printed to the console."]
--------------------------------------------------
```
