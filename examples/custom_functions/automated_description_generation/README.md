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

The documentation will also cover configuration considerations and how to set up a NeMo Agent toolkit pipeline that leverages the workflow. The current implementation is Milvus focused, with a plans to extend functionality to other vector databases.

## Table of Contents

* [Key Features](#key-features)
* [Installation and Setup](#installation-and-setup)
  * [Install this Workflow](#install-this-workflow)
  * [Set Up API Keys](#set-up-api-keys)
  * [Set Up Milvus](#set-up-milvus)
  * [Bootstrap Data](#bootstrap-data)
* [Example Usage](#example-usage)
  * [No Automated Description Generation](#no-automated-description-generation)
  * [Automated Description Generation](#automated-description-generation)


## Key Features

- **VectorDB Collection Analysis:** Demonstrates automated generation of intelligent descriptions for VectorDB collections using document retrieval and LLM-based summarization to capture the essence of stored documents.
- **Multi-Query Collection Enhancement:** Shows how to enhance multi-query collection workflows by automatically generating feature-rich descriptions that improve retrieval accuracy across multiple collections.
- **Map-Reduce Summarization:** Implements a sophisticated approach using dummy embeddings for document retrieval, LLM-generated local summaries, and map-reduce techniques for final description generation.
- **Milvus Integration with Extensible Design:** Currently focused on Milvus vector database with plans for extension to other VectorDBs, demonstrating how to work with the NeMo Agent toolkit retriever interface.
- **RAG Service Enhancement:** Provides a foundation for improving RAG (Retrieval-Augmented Generation) services by automatically generating more accurate collection metadata for better document retrieval.

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e ./examples/custom_functions/automated_description_generation
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

### Set Up Milvus

This example uses a Milvus vector database to demonstrate how descriptions can be generated for collections. However, because this workflow uses the built-in NeMo Agent toolkit abstractions for retrievers, this example will work for any database that implements the required methods of the NeMo Agent toolkit `retriever` interface.

Start the docker compose
```bash
docker compose -f examples/deploy/docker-compose.milvus.yml up -d
```

> [!NOTE]
> It can take some time for Milvus to start up. You can check the logs with:
> ```bash
> docker compose -f examples/deploy/docker-compose.milvus.yml logs --follow
> ```

### Bootstrap Data

To use this example, you will also need to create a `wikipedia_docs` and a `cuda_docs` collection in your Milvus database. The following script will create the collections and populate the data:

```bash
python scripts/langchain_web_ingest.py --collection_name=cuda_docs
python scripts/langchain_web_ingest.py --urls https://en.wikipedia.org/wiki/Aardvark --collection_name=wikipedia_docs
```

## Example Usage

### No Automated Description Generation

To demonstrate the benefit of this methodology to automatically generate collection descriptions, we will use it in a function that can automatically discover and generate descriptions for collections within a given vector database.
It will then rename the retriever tool for that database with the generated description instead of the user-provided description. Let us explore the `config_no_auto.yml` file, that performs simple RAG.

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 10000

embedders:
  milvus_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5
    temperature: 0.0
    truncate: "END"

retrievers:
  retriever:
    _type: milvus_retriever
    uri: http://localhost:19530
    collection_name: wikipedia_docs
    embedding_model: milvus_embedder
    top_k: 10

functions:
  cuda_tool:
    _type: nat_retriever
    retriever: retriever
    # Intentionally mislabelled to show the effects of poor descriptions
    topic: NVIDIA CUDA
    description: This tool can only retrieve information about NVIDIA's CUDA library.

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
nat run --config_file examples/custom_functions/automated_description_generation/configs/config_no_auto.yml --input "List all known subspecies of Aardvark"
```

**Expected Workflow Output**
```console
2025-10-17 11:35:32 - INFO     - nat.cli.commands.start:192 - Starting NAT from config file: 'examples/custom_functions/automated_description_generation/configs/config_no_auto.yml'
2025-10-17 11:35:32 - INFO     - nat.retriever.milvus.retriever:63 - Mivlus Retriever using _search for search.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 1
Number of Function Groups: 0
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 1
Number of TTC Strategies: 0
Number of Authentication Providers: 0

2025-10-17 11:35:33 - INFO     - nat.agent.react_agent.agent:169 - 
------------------------------
[AGENT]
Agent input: List all known subspecies of Aardvark
Agent's thoughts: 
Thought: The previous conversation history does not provide any information about the subspecies of Aardvark. I should start from scratch to answer this question.


Action: None
Action Input: None

------------------------------
2025-10-17 11:35:33 - WARNING  - nat.agent.react_agent.agent:273 - [AGENT] ReAct Agent wants to call tool None. In the ReAct Agent's configuration within the config file,there is no tool with that name: ['cuda_tool']
2025-10-17 11:35:34 - INFO     - nat.agent.react_agent.agent:193 - 
------------------------------
[AGENT]
Agent input: List all known subspecies of Aardvark
Agent's thoughts: 
Thought: Since there is no tool available to provide information about Aardvark subspecies, I should inform the user that I'm unable to answer the question.


Final Answer: Unfortunately, I'm unable to provide information about Aardvark subspecies as it is not within my knowledge domain or available tools.
------------------------------
2025-10-17 11:35:34 - WARNING  - nat.builder.intermediate_step_manager:94 - Step id 4de1cd41-bd02-4b05-9478-4388922f7d00 not found in outstanding start steps
2025-10-17 11:35:34 - INFO     - nat.front_ends.console.console_front_end_plugin:102 - --------------------------------------------------
Workflow Result:
["Unfortunately, I'm unable to provide information about Aardvark subspecies as it is not within my knowledge domain or available tools."]
--------------------------------------------------
```

If we look at the full output from the toolkit, we see that the agent did not call the tool for retrieval as it was incorrectly described.

### Automated Description Generation

Let us see what happens if we use the automated description generate function to intelligently sample the documents in the retriever and create an appropriate description. We could do so with the following configuration:

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 10000

embedders:
  milvus_embedder:
    _type: nim
    model_name: nvidia/nv-embedqa-e5-v5
    temperature: 0.0
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
    _type: nat_retriever
    retriever: retriever
    # Intentionally mislabelled to show the effects of poor descriptions
    topic: NVIDIA CUDA
    description: This tool can only retrieve information about NVIDIA's CUDA library.
  retrieve_tool:
    _type: automated_description_milvus
    llm_name: nim_llm
    retriever_name: retriever
    retrieval_tool_name: cuda_tool
    collection_name: wikipedia_docs

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
nat run --config_file examples/custom_functions/automated_description_generation/configs/config.yml --input "List all known subspecies of Aardvark"
```

**Expected Workflow Output**
```console
2025-10-17 11:36:41 - INFO     - nat.cli.commands.start:192 - Starting NAT from config file: 'examples/custom_functions/automated_description_generation/configs/config.yml'
2025-10-17 11:36:41 - INFO     - nat.retriever.milvus.retriever:63 - Mivlus Retriever using _search for search.
2025-10-17 11:36:41 - INFO     - nat_automated_description_generation.register:61 - Building necessary components for the Automated Description Generation Workflow
2025-10-17 11:36:41 - INFO     - nat_automated_description_generation.register:72 - Components built, starting the Automated Description Generation Workflow
2025-10-17 11:36:44 - INFO     - nat_automated_description_generation.register:87 - Generated the dynamic description: Ask questions about the following collection of text: This collection appears to be a comprehensive repository of information on the aardvark, storing a wide range of data types including text, images, and taxonomic classifications, with the primary purpose of providing a detailed and authoritative reference on the biology, behavior, and conservation of the aardvark species.

Configuration Summary:
--------------------
Workflow Type: react_agent
Number of Functions: 2
Number of Function Groups: 0
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Object Stores: 0
Number of Retrievers: 1
Number of TTC Strategies: 0
Number of Authentication Providers: 0

2025-10-17 11:36:45 - INFO     - nat.agent.react_agent.agent:169 - 
------------------------------
[AGENT]
Agent input: List all known subspecies of Aardvark
Agent's thoughts: 
Thought: I need to find information about the subspecies of Aardvark.

Action: retrieve_tool
Action Input: {'query': 'What are the known subspecies of Aardvark?'}


------------------------------
2025-10-17 11:36:46 - INFO     - nat.tool.retriever:76 - Retrieved 10 records for query What are the known subspecies of Aardvark?.
2025-10-17 11:36:46 - INFO     - nat.agent.base:221 - 
------------------------------
[AGENT]
Calling tools: retrieve_tool
Tool's input: {'query': 'What are the known subspecies of Aardvark?'}
Tool's response: 
{"results": [{"page_content": "Subspecies[edit]\nThe aardvark has seventeen poorly defined subspecies listed:[4]\n\nOrycteropus afer afer (Southern aardvark)\nO. a. adametzi  Grote, 1921 (Western aardvark)\nO. a. aethiopicus  Sundevall, 1843\nO. a. angolensis  Zukowsky & Haltenorth, 1957\nO. a. erikssoni  L\u00f6nnberg, 1906\nO. a. faradjius  Hatt, 1932\nO. a. haussanus  Matschie, 1900\nO. a. kordofanicus  Rothschild, 1927\nO. a. lademanni  Grote, 1911\nO. a. leptodon  Hirst, 1906\nO. a. matschiei  Grote, 1921\nO. a. observandus Grote, 1921\nO. a. ruvanensis Grote, 1921\nO. a. senegalensis Lesson, 1840\nO. a. somalicus Lydekker, 1908\nO. a. wardi Lydekker, 1908\nO. a. wertheri  Matschie, 1898 (Eastern aardvark)\nThe 1911 Encyclop\u00e6dia Britannica also mentions O.\u00a0a. capensis or Cape ant-bear from South Africa.[21]\n\nDescription[edit]\nSouthern aardvark (O.\u00a0a. afer) front and rear foot print\nStrong forelimb of aardvark\nThe aardvark is vaguely pig-like in appearance. Its ...(rest of response truncated)
------------------------------
2025-10-17 11:36:51 - INFO     - nat.agent.react_agent.agent:193 - 
------------------------------
[AGENT]
Agent input: List all known subspecies of Aardvark
Agent's thoughts: 
Thought: I have found the relevant information about the subspecies of Aardvark.

Final Answer: The aardvark has seventeen poorly defined subspecies listed, including Orycteropus afer afer, O. a. adametzi, O. a. aethiopicus, O. a. angolensis, O. a. erikssoni, O. a. faradjius, O. a. haussanus, O. a. kordofanicus, O. a. lademanni, O. a. leptodon, O. a. matschiei, O. a. observandus, O. a. ruvanensis, O. a. senegalensis, O. a. somalicus, O. a. wardi, and O. a. wertheri.
------------------------------
2025-10-17 11:36:51 - WARNING  - nat.builder.intermediate_step_manager:94 - Step id 327b094d-f883-47ab-837e-eca0a91ca557 not found in outstanding start steps
2025-10-17 11:36:51 - INFO     - nat.front_ends.console.console_front_end_plugin:102 - --------------------------------------------------
Workflow Result:
['The aardvark has seventeen poorly defined subspecies listed, including Orycteropus afer afer, O. a. adametzi, O. a. aethiopicus, O. a. angolensis, O. a. erikssoni, O. a. faradjius, O. a. haussanus, O. a. kordofanicus, O. a. lademanni, O. a. leptodon, O. a. matschiei, O. a. observandus, O. a. ruvanensis, O. a. senegalensis, O. a. somalicus, O. a. wardi, and O. a. wertheri.']
--------------------------------------------------
```

There are two key differences in the workflow execution:

1. The generated description correctly reflected the contents of the collection.

    > Generated the dynamic description: Ask questions about the following collection of text: This collection appears to be a comprehensive repository of information on the aardvark, storing a wide range of data types including text, images, and taxonomic classifications, with the primary purpose of providing a detailed and authoritative reference on the biology, behavior, and conservation of the aardvark species.

2. We see that the agent called the `retrieve_tool`.

This example demonstrates how the automated description generation tool can be used to automatically generate descriptions for collections within a vector database. While this is a toy example, this can be quite helpful when descriptions are vague, or you have too many collections to describe!
