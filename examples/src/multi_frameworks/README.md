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

# Multi-Frameworks Example

This example demonstrates how to integrate multiple AI frameworks seamlessly using a set of LangChain / LangGraph agents, in AIQ toolkit.
AIQ toolkit is framework-agnostic, allowing usage of custom and pre-built preferred AI tools without restriction due to AI framework.

## Overview

LangChain is incredibly flexible, LlamaIndex is incredibly powerful for building RAG pipelines;
different AI frameworks excel at different tasks.
Instead of committing to just one, this example shows how they can work together via AIQ toolkit.

In this example, we combine:
- **Haystack Agent** – with a configurable LLM.
- **LangChain Research Tool** – web search.
- **LlamaIndex RAG Tool** – document Q&A (pre-configured to use this README)

This example workflow leverages the AIQ toolkit plugin system and `Builder` object to demonstrate how the `Builder` object can dynamically wrap any Python function—regardless of its underlying AI framework or implementation—and convert it into another AI framework of our choice.

In this example, we wrap all three of the above tools as LangChain Tools.
Then, using LangChain and LangGraph, we unify these frameworks into a single workflow, demonstrating interoperability and flexibility. The goal is not to favor one tool over another but to showcase how different AI stacks can complement each other.


## Why This Matters

- **Leverage Strengths** – Different AI frameworks specialize in different areas.
- **Interoperability** – Combine tools seamlessly without vendor lock-in.
- **Scalability** – Build flexible AI pipelines that adapt to different use cases.


## Key Features

- **Custom-plug-in Tools:** with a basic llama-index RAG ingesting README from within this workflow
- **Custom Plugin System:** Developers can bring in new tools using plugins.
- **High-level API:** Enables defining functions that transform into asynchronous LangChain tools.
- **Agentic Workflows:** Fully configurable via YAML for flexibility and productivity.
- **Ease of Use:** Simplifies developer experience and deployment.

There is a supervisor agent that will assign/route incoming user query to one of the worker agents.
the 3 worker agents are :

- (1) a `rag_agent` made out of `llama_index` via a custom `llama-index-rag` tool
- (2) a `research_agent` made out of a LangChain runnable chain with tool calling capability, able to call arXiv as a tool and return summarized found research papers
- (3) a chitchat agent that is able to handle general chitchat query from user, constructed via haystack's pipeline

the multi-agents architecture looks like the below

![LangGraph multi-agents workflow](../../../../docs/source/_static/aiq_multi_frameworks_agentic_schema.png)

## Local Installation and Usage

If you have not already done so, follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Step 1: Set Your NVIDIA API Key and Tavily API Key Environment Variable
If you have not already done so, follow the [Obtaining API Keys](../../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services.

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

For Tavily API key, create an account at [`tavily.com`](https://tavily.com/) and obtain an API key. Once obtained, set the `TAVILY_API_KEY` environment variable to the API key:
```bash
export TAVILY_API_KEY=<YOUR_TAVILY_API_KEY>
```

### Step 2: Running the `multi_frameworks` Workflow

**Install the `multi_frameworks` Workflow**

```bash
uv pip install -e examples/basic/frameworks/multi_frameworks
```

**Run the `multi_frameworks` Workflow**

note: the below is an example command to use and query this and trigger `rag_agent`

```bash
aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "tell me about this workflow"
```
**expected output:**

```
(.venv) (base) coder ➜ ~/dev/ai-query-engine $ aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "tell me about this workflow"
/home/coder/dev/ai-query-engine/.venv/lib/python3.12/site-packages/pydantic/_internal/_config.py:341: UserWarning: Valid config keys have changed in V2:
* 'allow_population_by_field_name' has been renamed to 'populate_by_name'
  warnings.warn(message, UserWarning)
2025-01-16 18:53:33,577 - aiq.cli.run - INFO - Loading configuration from: examples/basic/frameworks/multi_frameworks/configs/config.yml
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
##### processing data from ingesting files in this folder : /home/coder/dev/ai-query-engine/examples/basic/frameworks/multi_frameworks/data/README.md
2025-01-16 18:53:37,559 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
/opt/conda/lib/python3.12/contextlib.py:210: LangChainDeprecationWarning: As of langchain-core 0.3.0, LangChain uses pydantic v2 internally. The langchain_core.pydantic_v1 module was a compatibility shim for pydantic v1, and should no longer be used. Please update the code to import from Pydantic directly.

For example, replace imports like: `from langchain_core.pydantic_v1 import BaseModel`
with: `from pydantic import BaseModel`
or the v1 compatibility namespace if you are working in a code base that has not been fully upgraded to pydantic 2 yet.         from pydantic.v1 import BaseModel

  return await anext(self.gen)
workflow config =  llm_name='meta/llama-3.1-405b-instruct' llm='nim_llm' embedding_name='nvidia/nv-embed-v1' tool_names=['llama_index_rag'] data_dir='/home/coder/dev/ai-query-engine/examples/basic/frameworks/multi_frameworks/data/README.md'
 <llama_index.core.tools.function_tool.FunctionTool object at 0x7f5d8552b290> <class 'llama_index.core.tools.function_tool.FunctionTool'>

Configuration Summary:
--------------------
Workflow Type: multi_frameworks
Number of Functions: 1
Number of LLMs: 1
Number of Embedders: 0
Number of Memory: 0

2025-01-16 18:53:39,550 - aiq.cli.run - INFO - Processing input: ('tell me about this workflow',)
========== inside **supervisor node**  current status =
 {'input': 'tell me about this workflow', 'chat_history': []}
========== inside **router node**  current status =
 ['input', 'chosen_worker_agent', 'chat_history']
 ############# router to --> workers
========== inside **workers node**  current status =
 {'input': 'tell me about this workflow', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='tell me about this workflow', additional_kwargs={}, response_metadata={}), AIMessage(content='Retrieve', additional_kwargs={}, response_metadata={})]), 'chosen_worker_agent': 'Retrieve'}
 <class 'llama_index.core.query_engine.retriever_query_engine.RetrieverQueryEngine'> <llama_index.core.query_engine.retriever_query_engine.RetrieverQueryEngine object at 0x7f5d855282c0> tell me about this workflow
2025-01-16 18:53:41,528 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/embeddings "HTTP/1.1 200 OK"
2025-01-16 18:53:45,816 - httpx - INFO - HTTP Request: POST https://integrate.api.nvidia.com/v1/chat/completions "HTTP/1.1 200 OK"
 **using rag_tool via llama_index_rag_tool >>> output:
 This workflow is a multi-frameworks example that can be installed locally and run using specific commands. To install the workflow, you need to run `uv pip install -e examples/basic/frameworks/multi_frameworks`. After installation, you can run the workflow using the command `aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "your query here"`. You can replace "your query here" with any input you want to query the workflow with.
 <class 'str'> This workflow is a multi-frameworks example that can be installed locally and run using specific commands. To install the workflow, you need to run `uv pip install -e examples/basic/frameworks/multi_frameworks`. After installation, you can run the workflow using the command `aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "your query here"`. You can replace "your query here" with any input you want to query the workflow with.
2025-01-16 18:53:45,821 - aiq.cli.run - INFO - --------------------------------------------------
Workflow Result:
['This workflow is a multi-frameworks example that can be installed locally and run using specific commands. To install the workflow, you need to run `uv pip install -e examples/basic/frameworks/multi_frameworks`. After installation, you can run the workflow using the command `aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "your query here"`. You can replace "your query here" with any input you want to query the workflow with.']
--------------------------------------------------
Cleaning up multi_frameworks workflow.
2025-01-16 18:53:45,822 - aiq.cli.entrypoint - INFO - Total time: 12.25 sec
2025-01-16 18:53:45,823 - aiq.cli.entrypoin
```
note: the below is an example command to use and query this and trigger `research_agent`

```bash
aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "what is RAG?"
```
**expected output:**
```
(.venv) AgentIQ % aiq run --config_file=examples/basic/frameworks/multi_frameworks/configs/config.yml --input "what is RAG?"
2025-05-14 15:19:32,924 - aiq.runtime.loader - WARNING - Loading module 'aiq_profiler_agent.register' from entry point 'aiq_profiler_agent' took a long time (1747.276783 ms). Ensure all imports are inside your registered functions.
2025-05-14 15:19:33,092 - aiq.runtime.loader - WARNING - Loading module 'aiq.plugins.agno.register' from entry point 'aiq_agno' took a long time (141.694069 ms). Ensure all imports are inside your registered functions.
2025-05-14 15:19:33,305 - aiq.runtime.loader - WARNING - Loading module 'aiq_multi_frameworks.register' from entry point 'aiq_multi_frameworks' took a long time (212.839842 ms). Ensure all imports are inside your registered functions.
2025-05-14 15:19:33,848 - aiq.runtime.loader - WARNING - Loading module 'aiq_alert_triage_agent.register' from entry point 'aiq_alert_triage_agent' took a long time (303.922176 ms). Ensure all imports are inside your registered functions.
2025-05-14 15:19:34,080 - aiq.cli.commands.start - INFO - Starting AIQ Toolkit from config file: 'examples/basic/frameworks/multi_frameworks/configs/config.yml'
2025-05-14 15:19:34,082 - aiq.cli.commands.start - WARNING - The front end type in the config file (fastapi) does not match the command name (console). Overwriting the config file front end.
2025-05-14 15:19:41,048 - aiq_multi_frameworks.llama_index_rag_tool - INFO - ##### processing data from ingesting files in this folder : ./examples/basic/frameworks/multi_frameworks/README.md
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
2025-05-14 15:19:51,208 - aiq_multi_frameworks.register - INFO - workflow config = llm='nim_llm' data_dir='./examples/basic/frameworks/multi_frameworks/README.md' research_tool='langchain_researcher_tool' rag_tool='llama_index_rag' chitchat_agent='haystack_chitchat_agent'

Configuration Summary:
--------------------
Workflow Type: multi_frameworks
Number of Functions: 4
Number of LLMs: 1
Number of Embedders: 1
Number of Memory: 0
Number of Retrievers: 0

2025-05-14 15:19:51,563 - aiq_multi_frameworks.register - INFO - ========== inside **supervisor node**  current status =
 {'input': 'what is RAG?', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='what is RAG?', additional_kwargs={}, response_metadata={}), AIMessage(content='Research', additional_kwargs={}, response_metadata={})])}
2025-05-14 15:19:51,564 - aiq_multi_frameworks.register - INFO - ========== inside **router node**  current status =
 , ['input', 'chosen_worker_agent', 'chat_history']
2025-05-14 15:19:51,564 - aiq_multi_frameworks.register - INFO -  ############# router to --> workers
2025-05-14 15:19:51,564 - aiq_multi_frameworks.register - INFO - ========== inside **workers node**  current status =
 , {'input': 'what is RAG?', 'chat_history': InMemoryChatMessageHistory(messages=[HumanMessage(content='what is RAG?', additional_kwargs={}, response_metadata={}), AIMessage(content='Research', additional_kwargs={}, response_metadata={})]), 'chosen_worker_agent': 'Research'}
2025-05-14 15:19:54,119 - aiq_multi_frameworks.langchain_research_tool - INFO - output from langchain_research_tool: Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific
2025-05-14 15:19:54,121 - aiq_multi_frameworks.register - INFO - final_output : Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific
2025-05-14 15:19:54,121 - aiq.front_ends.console.console_front_end_plugin - INFO -
--------------------------------------------------
Workflow Result:
['Retrieval-Augmented Generation (RAG) is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific']
--------------------------------------------------
```
