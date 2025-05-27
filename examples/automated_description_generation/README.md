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

<!--
  SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->

# Automated Description Generation Workflow

The automated description generation workflow, is a workflow that can be used to build on top of the RAG service and enhances the accuracy of the  multi-query collection workflow. The goal of the workflow is to automatically generate descriptions of collections within VectorDB's, which can be leveraged by the multi-query collection tool to empower retrieval of context, typically documents, across multiple collections within a given vector database. This document will cover the tooling and the process leveraged to execute the description generation workflow.

The documentation will also cover configuration considerations and how to set up an AIQ toolkit pipeline that leverages the workflow. The current implementation is Milvus focused, with a plans to extend functionality to other vector databases.

## Table of Contents

* [Key Features](#key-features)
* [Installation and Usage](#installation-and-setup)
* [Example Usage](#example-usage)


## Key Features

The automated description generation workflow is responsible for intelligently generating descriptions from collections within a given VectorDB. This is useful for generating feature rich descriptions that are representative of the documents present within a given collection, reducing the need for human generated descriptions which may not fully capture general nature of the collection. The workflow is able to achieve this by performing the following steps:

1. Take an input collection name - the collection is expected to be present within the VectorDB with documents already ingested.
2. Using a dummy embedding vector, perform retrieval and return the top K entries within the target collection.
3. Using retrieved documents, an LLM is used to generate a set of local summaries.
4. Using an LLM and a map reduce approach, the local summaries are leveraged to generate a final description for the target collection.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Install this Workflow:

From the root directory of the AIQ toolkit library, run the following commands:

```bash
uv pip install -e ./examples/automated_description_generation
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Setting Up Milvus

This example uses a Milvus vector database to demonstrate how descriptions can be generated for collections. However, because this workflow uses the built-in AIQ toolkit abstractions for retrievers, this example will work for any database that implements the required methods of the AIQ toolkit `retriever` interface.

The rest of this example assumes you have a running instance of Milvus at `localhost:19530`. If you would like a guide on setting up the database used in this example, please follow
the instructions in the `simple_rag` example of AIQ toolkit [here](../simple_rag/README.md).

If you have a different Milvus database you would like to use, please modify the `./configs/config.yml` with the appropriate URLs to your database instance.

To use this example, you will also need to create a `wikipedia_docs` and a `cuda_docs` collection in your Milvus database. You can do this by following the instructions in the `simple_rag` example of AIQ toolkit [here](../simple_rag/README.md) and running the following command:

```bash
python scripts/langchain_web_ingest.py
python scripts/langchain_web_ingest.py --urls https://en.wikipedia.org/wiki/Aardvark --collection_name=wikipedia_docs
```
## Example Usage

To demonstrate the benefit of this methodology to automatically generate collection descriptions, we will use it in a function that can automatically discover and generate descriptions for collections within a given vector database.
It will then rename the retriever tool for that database with the generated description instead of the user-provided description. Let us explore the `config_no_auto.yml` file, that performs simple RAG.

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    temperature: 0.0
    max_tokens: 10000

embedders:
  milvus_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5
    truncate: "END"

retrievers:
  retriever:
    _type: milvus_retriever
    uri: http://localhost:19530
    collection_name: "wikipedia_docs"
    embedding_model: milvus_embedder
    top_k: 10

functions:
  cuda_tool:
    _type: aiq_retriever
    retriever: retriever
    # Intentionally mislabelled to show the effects of poor descriptions
    topic: NVIDIA CUDA
    description: Only to search about NVIDIA CUDA

workflow:
  _type: react_agent
  tool_names:
   - cuda_tool
  verbose: true
  llm_name: nim_llm
```

Like in the `simple_rag` example, we demonstrate the use of the `react_agent` tool to execute the workflow. The `react_agent` tool will execute workflow with the given function. However, you have noticed that the `cuda_tool` is incorrectly named and labelled! it points to a retriever that contains documents
from Wikipedia, but the agent may not know that because the description is inaccurate.

Let us explore the output of running the agent without an automated description generation tool:

```bash
aiq run --config_file examples/automated_description_generation/configs/config_no_auto.yml --input "List 5 subspecies of Aardvark?"
```

The expected output is as follows:

```console
2025-03-14 06:23:47,362 - aiq.front_ends.console.console_front_end_plugin - INFO - Processing input: ('List 5 subspecies of Aardvark?',)
2025-03-14 06:23:47,365 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-14 06:23:48,266 - aiq.agent.react_agent.agent - INFO - The user's question was: List 5 subspecis of Aardvark?
2025-03-14 06:23:48,267 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To answer this question, I need to find information about the subspecies of Aardvark. I will use my knowledge database to find the answer.

Action: None
Action Input: None


2025-03-14 06:23:48,271 - aiq.agent.react_agent.agent - WARNING - ReAct Agent wants to call tool None. In the ReAct Agent's configuration within the config file,there is no tool with that name: ['cuda_tool']
2025-03-14 06:23:48,273 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-14 06:23:49,755 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
You are correct, there is no tool named "None". Since the question is about Aardvark subspecies and not related to NVIDIA CUDA, I should not use the cuda_tool.

Instead, I will provide a general answer based on my knowledge.

Thought: I now know the final answer
Final Answer: There is only one species of Aardvark, Orycteropus afer, and it has no recognized subspecies.
2025-03-14 06:23:49,758 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-14 06:23:49,758 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['There is only one species of Aardvark, Orycteropus afer, and it has no recognized subspecies.']
--------------------------------------------------
```

We see that the agent did not call tool for retrieval as it was incorrectly described. However, let us see what happens if we use the automated description generate function to intelligently sample the documents in the retriever and create an appropriate description. We could do so with the following configuration:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    temperature: 0.0
    max_tokens: 10000

embedders:
  milvus_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5
    truncate: "END"

retrievers:
  retriever:
    _type: milvus_retriever
    uri: http://localhost:19530
    collection_name: "wikipedia_docs"
    embedding_model: milvus_embedder
    top_k: 10

functions:
  cuda_tool:
    _type: aiq_retriever
    retriever: retriever
    # Intentionally mislabelled to show the effects of poor descriptions
    topic: NVIDIA CUDA
    description: This tool retrieves information about NVIDIA's CUDA library
  retrieve_tool:
    _type: automated_description_milvus
    llm_name: nim_llm
    retriever_name: retriever
    retrieval_tool_name: cuda_tool
    collection_name: cuda_docs

workflow:
  _type: react_agent
  tool_names:
   - retrieve_tool
  verbose: true
  llm_name: nim_llm
```
Here, we're searching for information about Wikipedia in a collection using a tool incorrectly described to contain documents about NVIDIA's CUDA library. We see above that we use the automated description generation tool to generate a description for the collection `wikipedia_docs`. The tool uses the `retriever` to retrieve documents from the collection, and then uses the `nim_llm` to generate a description for the collection.

If we run the updated configuration, we see the following output:

```bash
aiq run --config_file examples/automated_description_generation/configs/config.yml --input "List 5 subspecies of Aardvark?"
```

The expected output is as follows:

```console
$ aiq run --config_file examples/automated_description_generation/configs/config.yml --input "List 5 subspecies of Aardvark?"
2025-05-16 11:07:32,969 - aiq.runtime.loader - WARNING - Loading module 'aiq_profiler_agent.register' from entry point 'aiq_profiler_agent' took a long time (317.265034 ms). Ensure all imports are inside your registered functions.
2025-05-16 11:07:33,468 - aiq.runtime.loader - WARNING - Loading module 'aiq.agent.register' from entry point 'aiq_agents' took a long time (366.579533 ms). Ensure all imports are inside your registered functions.
2025-05-16 11:07:33,675 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/automated_description_generation/configs/config.yml'
2025-05-16 11:07:33,687 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-05-16 11:07:33,898 - aiq.retriever.milvus.retriever - INFO - Mivlus Retriever using _search for search.
2025-05-16 11:07:33,900 - aiq_automated_description_generation.register - INFO - Building necessary components for the Automated Description Generation Workflow
2025-05-16 11:07:33,928 - aiq_automated_description_generation.register - INFO - Components built, starting the Automated Description Generation Workflow
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
Token indices sequence length is longer than the specified maximum sequence length for this model (1253 > 1024). Running this sequence through the model will result in indexing errors
2025-05-16 11:07:37,298 - aiq_automated_description_generation.register - INFO - Generated the dynamic description: Ask questions about the following collection of text: This collection appears to be a technical documentation of NVIDIA's CUDA toolkit, storing various code snippets, instructions, and guidelines for optimizing GPU-accelerated applications, with the primary purpose of assisting developers in harnessing the power of NVIDIA GPUs for general-purpose computing.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 1

2025-05-16 11:07:40,778 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: List 5 subspecies of Aardvark?
Agent's thoughts:
Thought: The input question is asking for subspecies of Aardvark, but the provided text is about NVIDIA's CUDA toolkit, which is unrelated to Aardvarks. I will ask the human to use a tool to find the answer.

Action: retrieve_tool
Action Input: None


------------------------------
2025-05-16 11:07:41,012 - aiq.tool.retriever - INFO - Retrieved 10 records for query None.
2025-05-16 11:07:41,014 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: retrieve_tool
Tool's input: None
Tool's response:
{"results": [{"page_content": "This means, in particular, that a host thread using the runtime API without explicitly calling cudaSetDevice() might be associated with a device other than device 0 if device 0 turns out to be in prohibited mode or in exclusive-process mode and used by another process. cudaSetValidDevices() can be used to set a device from a prioritized list of devices.\nNote also that, for devices featuring the Pascal architecture onwards (compute capability with major revision number 6 and higher), there exists support for Compute Preemption. This allows compute tasks to be preempted at instruction-level granularity, rather than thread block granularity as in prior Maxwell and Kepler GPU architecture, with the benefit that applications with long-running kernels can be prevented from either monopolizing the system or timing out. However, there will be context switch overheads associated with Compute Preemption, which is automatically enabled on those devices for which su...
------------------------------
2025-05-16 11:07:44,801 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: List 5 subspecies of Aardvark?
Agent's thoughts:
Thought: The provided tool output does not contain any information about Aardvark subspecies. The output appears to be related to NVIDIA's CUDA toolkit and does not mention Aardvarks at all.

Action: None
Action Input: None

Thought: Since the provided tool output does not contain any relevant information, I will provide a final answer based on general knowledge.

Final Answer: Unfortunately, I couldn't find any information about Aardvark subspecies in the provided text. However, according to general knowledge, there are no recognized subspecies of Aardvarks. Aardvarks are a single species (Orycteropus afer) and do not have any subspecies.
------------------------------
2025-05-16 11:07:44,802 - aiq.agent.react_agent.agent - WARNING - [AGENT] Error parsing agent output
Observation:Parsing LLM output produced both a final answer and a parse-able action:: Thought: The provided tool output does not contain any information about Aardvark subspecies. The output appears to be related to NVIDIA's CUDA toolkit and does not mention Aardvarks at all.

Action: None
Action Input: None

Thought: Since the provided tool output does not contain any relevant information, I will provide a final answer based on general knowledge.

Final Answer: Unfortunately, I couldn't find any information about Aardvark subspecies in the provided text. However, according to general knowledge, there are no recognized subspecies of Aardvarks. Aardvarks are a single species (Orycteropus afer) and do not have any subspecies.
Agent Output:
Thought: The provided tool output does not contain any information about Aardvark subspecies. The output appears to be related to NVIDIA's CUDA toolkit and does not mention Aardvarks at all.

Action: None
Action Input: None

Thought: Since the provided tool output does not contain any relevant information, I will provide a final answer based on general knowledge.

Final Answer: Unfortunately, I couldn't find any information about Aardvark subspecies in the provided text. However, according to general knowledge, there are no recognized subspecies of Aardvarks. Aardvarks are a single species (Orycteropus afer) and do not have any subspecies.
2025-05-16 11:07:44,802 - aiq.agent.react_agent.agent - INFO - [AGENT] Retrying ReAct Agent, including output parsing Observation
2025-05-16 11:07:48,755 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: List 5 subspecies of Aardvark?
Agent's thoughts:
Thought: The input question is asking for subspecies of Aardvark, but the provided text is about NVIDIA's CUDA toolkit, which is unrelated to Aardvarks. I will ask the human to use a tool to find the answer.

Action: retrieve_tool
Action Input: {"query": "Aardvark subspecies"}
------------------------------
2025-05-16 11:07:48,993 - aiq.tool.retriever - INFO - Retrieved 10 records for query Aardvark subspecies.
2025-05-16 11:07:48,995 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: retrieve_tool
Tool's input: {"query": "Aardvark subspecies"}
Tool's response:
{"results": [{"page_content": "Subspecies\nThe aardvark has seventeen poorly defined subspecies listed:[4]\n\nOrycteropus afer afer (Southern aardvark)\nO. a. adametzi  Grote, 1921 (Western aardvark)\nO. a. aethiopicus  Sundevall, 1843\nO. a. angolensis  Zukowsky & Haltenorth, 1957\nO. a. erikssoni  L\u00f6nnberg, 1906\nO. a. faradjius  Hatt, 1932\nO. a. haussanus  Matschie, 1900\nO. a. kordofanicus  Rothschild, 1927\nO. a. lademanni  Grote, 1911\nO. a. leptodon  Hirst, 1906\nO. a. matschiei  Grote, 1921\nO. a. observandus Grote, 1921\nO. a. ruvanensis Grote, 1921\nO. a. senegalensis Lesson, 1840\nO. a. somalicus Lydekker, 1908\nO. a. wardi Lydekker, 1908\nO. a. wertheri  Matschie, 1898 (Eastern aardvark)\nThe 1911 Encyclop\u00e6dia Britannica also mentions O.\u00a0a. capensis or Cape ant-bear from South Africa.[21]\n\nDescription\nSouthern aardvark (O.\u00a0a. afer) front and rear foot print\nStrong forelimb of aardvark\nThe aardvark is vaguely pig-like in appearance. Its body is stou...
------------------------------
2025-05-16 11:07:51,650 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: List 5 subspecies of Aardvark?
Agent's thoughts:
Thought: The human has provided the results of the tool, which includes information about the aardvark's subspecies.

Action: None
Action Input: None


------------------------------
2025-05-16 11:07:51,651 - aiq.agent.react_agent.agent - WARNING - [AGENT] ReAct Agent wants to call tool None. In the ReAct Agent's configuration within the config file,there is no tool with that name: ['retrieve_tool']
2025-05-16 11:07:54,720 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: List 5 subspecies of Aardvark?
Agent's thoughts:
Thought: The human has provided the results of the tool, which includes information about the aardvark's subspecies.

Action: retrieve_tool
Action Input: {"query": "List 5 subspecies of Aardvark"}
------------------------------
2025-05-16 11:07:54,954 - aiq.tool.retriever - INFO - Retrieved 10 records for query List 5 subspecies of Aardvark.
2025-05-16 11:07:54,956 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Calling tools: retrieve_tool
Tool's input: {"query": "List 5 subspecies of Aardvark"}
Tool's response:
{"results": [{"page_content": "Subspecies\nThe aardvark has seventeen poorly defined subspecies listed:[4]\n\nOrycteropus afer afer (Southern aardvark)\nO. a. adametzi  Grote, 1921 (Western aardvark)\nO. a. aethiopicus  Sundevall, 1843\nO. a. angolensis  Zukowsky & Haltenorth, 1957\nO. a. erikssoni  L\u00f6nnberg, 1906\nO. a. faradjius  Hatt, 1932\nO. a. haussanus  Matschie, 1900\nO. a. kordofanicus  Rothschild, 1927\nO. a. lademanni  Grote, 1911\nO. a. leptodon  Hirst, 1906\nO. a. matschiei  Grote, 1921\nO. a. observandus Grote, 1921\nO. a. ruvanensis Grote, 1921\nO. a. senegalensis Lesson, 1840\nO. a. somalicus Lydekker, 1908\nO. a. wardi Lydekker, 1908\nO. a. wertheri  Matschie, 1898 (Eastern aardvark)\nThe 1911 Encyclop\u00e6dia Britannica also mentions O.\u00a0a. capensis or Cape ant-bear from South Africa.[21]\n\nDescription\nSouthern aardvark (O.\u00a0a. afer) front and rear foot print\nStrong forelimb of aardvark\nThe aardvark is vaguely pig-like in appearance. Its body is stou...
------------------------------
2025-05-16 11:08:02,632 - aiq.agent.react_agent.agent - INFO -
------------------------------
[AGENT]
Agent input: List 5 subspecies of Aardvark?
Agent's thoughts:
Thought: I now know the final answer

Final Answer: The 5 subspecies of Aardvark are:

1. Orycteropus afer afer (Southern aardvark)
2. O. a. adametzi  Grote, 1921 (Western aardvark)
3. O. a. aethiopicus  Sundevall, 1843
4. O. a. angolensis  Zukowsky & Haltenorth, 1957
5. O. a. erikssoni  Lönnberg, 1906
------------------------------
2025-05-16 11:08:02,634 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['The 5 subspecies of Aardvark are:\n\n1. Orycteropus afer afer (Southern aardvark)\n2. O. a. adametzi  Grote, 1921 (Western aardvark)\n3. O. a. aethiopicus  Sundevall, 1843\n4. O. a. angolensis  Zukowsky & Haltenorth, 1957\n5. O. a. erikssoni  Lönnberg, 1906']
--------------------------------------------------
```

We see that the agent called the `retrieve_tool`. This demonstrates how the automated description generation tool can be used to automatically generate descriptions for collections within a vector database. While this is a toy example, this can be quite helpful when descriptions are vague, or you have too many collections to describe!
