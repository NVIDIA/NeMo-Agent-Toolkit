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

The documentation will also cover configuration considerations and how to set up an AgentIQ pipeline that leverages the workflow. The current implementation is Milvus focused, with a plans to extend functionality to other vector databases.

## Table of Contents

* [Key Features](#key-features)
* [Installation and Usage](#installation-and-setup)
* [Example Usage](#example-usage)
* [Deployment-Oriented Setup](#deployment-oriented-setup)
   - [Build the Docker Image](#build-the-docker-image)
   - [Run the Docker Container](#run-the-docker-container)
   - [Test the API](#test-the-api)
   - [Expected API Output](#expected-api-output)

## Key Features

The automated description generation workflow is responsible for intelligently generating descriptions from collections within a given VectorDB. This is useful for generating feature rich descriptions that are representative of the documents present within a given collection, reducing the need for human generated descriptions which may not fully capture general nature of the collection. The workflow is able to achieve this by performing the following steps:

1. Take an input collection name - the collection is expected to be present within the vectorDB with documents already ingested.
2. Using a dummy embedding vector, perform retrieval and return the top K entries within the target collection.
3. Using retrieved documents, an LLM is used to generate a set of local summaries.
4. Using an LLM and a map reduce approach, the local summaries are leveraged to generate a final description for the target collection.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/intro/install.md) to create the development environment and install AgentIQ.

### Install this Workflow:

From the root directory of the AgentIQ library, run the following commands:

```bash
uv pip install -e ./examples/automated_description_generation
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../docs/source/intro/get-started.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Setting Up Milvus

This example uses a `Milvus` vector database to demonstrate how descriptions can be generated for collections. However, because this workflow uses AgentIQ's native abstractions
for retrievers, this example will work for any database that implements the required methods of the AgentIQ `retriever` interface.

The rest of this example assumes you have a running instance of Milvus at `localhost:19530`. If you would like a guide on setting up the database used in this example, please follow
the instructions in the `simple_rag` example of AgentIQ [here](../simple_rag/README.md).

If you have a different Milvus database you would like to use, please modify the `./configs/config.yml` with the appropriate URLs to your database instance.

To use this example, you will also need to create a `wikipedia_docs` collection in your Milvus database. You can do this by following the instructions in the `simple_rag` example of AgentIQ [here](../simple_rag/README.md) and running the following command:

```bash
python3 examples/simple_rag/ingestion/langchain_web_ingest.py --urls https://en.wikipedia.org/wiki/Aardvark --collection_name=wikipedia_docs
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

Like in the `simple_rag` example, we demonstrate the use of the `react_agent` tool to execute the workflow. The `react_agent` tool will execute workflow
with the given function. However, you have noticed that the `cuda_tool` is incorrectly named and labelled! it points to a retriever that contains documents
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

We see that the agent did not call tool for retrieval as it was incorrectly described. However, let us see what happens if we use the automated description generate function
to intelligently sample the documents in the retriever and create an appropriate description. We could do so with the following configuration:

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
Here, we're searching for information about Wikipedia in a collection using a tool incorrectly described to contain documents about NVIDIA's CUDA library.
We see above that we use the automated description generation tool to generate a description for the collection `wikipedia_docs`. The tool uses the `retriever` to retrieve documents from the collection, and then uses the `nim_llm` to generate a description for the collection.

If we run the updated configuration, we see the following output:

```bash
aiq run --config_file examples/automated_description_generation/configs/config.yml --input "List 5 subspecies of Aardvark?"
```

The expected output is as follows:

```console
## Omitted for brevity
Action: retrieve_tool
Action Input: {'query': 'Aardvark subspecies'}
2025-03-14 06:30:43,334 - aiq.agent.react_agent.agent - INFO - Calling tool retrieve_tool with input: {'query': 'Aardvark subspecies'}
2025-03-14 06:30:43,334 - aiq.agent.react_agent.agent - INFO - Successfully parsed structured tool input from Action Input
2025-03-14 06:30:43,759 - aiq.tool.retriever - INFO - Retrieved 10 records for query Aardvark subspecies.
2025-03-14 06:30:43,763 - aiq.agent.react_agent.agent - INFO - Querying agent, attempt: 1
2025-03-14 06:30:48,516 - aiq.agent.react_agent.agent - INFO -

The agent's thoughts are:
Thought: I now know the final answer
Final Answer: The 17 subspecies of Aardvark are: Orycteropus afer afer, O. a. adametzi, O. a. aethiopicus, O. a. angolensis, O. a. erikssoni, O. a. faradjius, O. a. haussanus, O. a. kordofanicus, O. a. lademanni, O. a. leptodon, O. a. matschiei, O. a. observandus, O. a. ruvanensis, O. a. senegalensis, O. a. somalicus, O. a. wardi, and O. a. wertheri.
2025-03-14 06:30:48,520 - aiq.observability.async_otel_listener - INFO - Intermediate step stream completed. No more events will arrive.
2025-03-14 06:30:48,520 - aiq.front_ends.console.console_front_end_plugin - INFO - --------------------------------------------------
Workflow Result:
['The 17 subspecies of Aardvark are: Orycteropus afer afer, O. a. adametzi, O. a. aethiopicus, O. a. angolensis, O. a. erikssoni, O. a. faradjius, O. a. haussanus, O. a. kordofanicus, O. a. lademanni, O. a. leptodon, O. a. matschiei, O. a. observandus, O. a. ruvanensis, O. a. senegalensis, O. a. somalicus, O. a. wardi, and O. a. wertheri.']
--------------------------------------------------
```

We see that the agent called the `retrieve_tool`. This demonstrates how the automated description generation tool can be used to automatically generate descriptions for collections within a vector database.
While this is a toy example, this can be quite helpful when descriptions are vague, or you have too many collections to describe!
