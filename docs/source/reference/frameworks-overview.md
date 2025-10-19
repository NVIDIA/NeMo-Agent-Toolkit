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

# Frameworks Overview

NVIDIA NeMo Agent toolkit (NAT) provides comprehensive support for multiple agentic frameworks, allowing you to use your preferred development tools while leveraging the capabilities of NAT. This document describes the framework integrations available and their respective levels of support.

## Supported Frameworks

NeMo Agent toolkit integrates with the following frameworks:

- **ADK**: Google Agent Development Kit for building AI agents
- **Agno**: A lightweight framework for building AI agents
- **CrewAI**: A framework for orchestrating role-playing AI agents
- **LangChain/LangGraph**: A framework for developing applications powered by large language models
- **LlamaIndex**: A data framework for building LLM applications
- **Semantic Kernel**: Microsoft's SDK for integrating LLMs with conventional programming languages

## Framework Support Levels

NeMo Agent toolkit provides different levels of support for each framework across the following dimensions:

### LLM Provider Support
The ability to use various large language model providers with a framework, including NVIDIA NIM, OpenAI, Azure OpenAI, AWS Bedrock, and LiteLLM.

### Embedder Provider Support
The ability to use embedding model providers for vector representations, including NVIDIA NIM embeddings, OpenAI embeddings, and Azure OpenAI embeddings.

### Retriever Provider Support
The ability to integrate with vector databases and retrieval systems, such as NeMo Retriever and Milvus.

### Tool Calling Support
The ability to use framework-specific tool calling mechanisms, allowing agents to invoke functions and tools during execution.

### Profiling Support
The ability to view workflow execution traces including intermediate steps, LLM calls, and tool calls within the NeMo Agent toolkit profiler.

## Framework Capabilities Matrix

The following table summarizes the current support level for each framework:

| Framework        | LLM Providers        | Embedder Providers     | Retriever Providers      | Tool Calling          | Profiling             |
|------------------|----------------------|------------------------|--------------------------|-----------------------|-----------------------|
| ADK              | ✅ Yes               | ❌ No                  | ❌ No                    | ✅ Yes                 | ✅ Yes                |
| Agno             | ⚠️ Limited           | ❌ No                  | ❌ No                    | ✅ Yes                 | ✅ Yes                |
| CrewAI           | ✅ Yes               | ❌ No                  | ❌ No                    | ✅ Yes                 | ✅ Yes                |
| LangChain        | ✅ Yes               | ✅ Yes                 | ✅ Yes                   | ✅ Yes                 | ✅ Yes                |
| LlamaIndex       | ✅ Yes               | ✅ Yes                 | ❌ No                    | ✅ Yes                 | ✅ Yes                |
| Semantic Kernel  | ⚠️ Limited           | ❌ No                  | ❌ No                    | ✅ Yes                 | ✅ Yes                |

## Framework-Specific Details

### ADK (Google Agent Development Kit)

Google's Agent Development Kit (ADK) is a framework for building AI agents with multiple LLM providers. It provides a set of tools for creating agents that can be used to create complex workflows powered by LLMs. ADK focuses on modularity and extensibility, making it suitable for integrating custom data pipelines and enhancing intelligent applications.

For more information, visit the [ADK website](https://google.github.io/adk-docs/).

| Capability              | Providers / Details                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **LLM Providers**       | NVIDIA NIM, OpenAI, Azure OpenAI, AWS Bedrock, LiteLLM                              |
| **Embedder Providers**  | None (use framework-agnostic embedders if needed)                                   |
| **Retriever Providers** | None (use ADK native tools)                                                         |
| **Tool Calling**        | Fully supported through the ADK `FunctionTool` interface                            |
| **Profiling**           | Comprehensive profiling support with instrumentation                                |

**Installation:**
```bash
uv pip install "nvidia-nat[adk]"
```

### Agno

Agno is a lightweight framework for building AI agents. It provides a set of tools for creating agents that can be used to create complex workflows powered by LLMs. Agno focuses on modularity and extensibility, making it suitable for integrating custom data pipelines and enhancing intelligent applications.

For more information, visit the [Agno website](https://agno.com/).

| Capability              | Providers / Details                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **LLM Providers**       | NVIDIA NIM, OpenAI, LiteLLM                                                         |
| **Embedder Providers**  | None (use framework-agnostic embedders if needed)                                   |
| **Retriever Providers** | None (use Agno native tools)                                                        |
| **Tool Calling**        | Fully supported through Agno's tool interface                                       |
| **Profiling**           | Comprehensive profiling support with instrumentation                                |

**Installation:**
```bash
uv pip install "nvidia-nat[agno]"
```

### CrewAI

CrewAI is a framework designed for orchestrating teams of role-playing AI agents that can collaborate and complete complex tasks. It enables the creation of agents with distinct roles, goals, and tools, allowing for multi-agent workflows adaptable to a wide range of scenarios—from research assistants to business process automation.

For more information, visit the [CrewAI website](https://www.crewai.com/).

| Capability              | Providers / Details                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **LLM Providers**       | NVIDIA NIM, OpenAI, Azure OpenAI, AWS Bedrock, LiteLLM                              |
| **Embedder Providers**  | None (use framework-agnostic embedders if needed)                                   |
| **Retriever Providers** | None (use CrewAI native tools)                                                      |
| **Tool Calling**        | Fully supported through CrewAI's tool system                                        |
| **Profiling**           | Comprehensive profiling support with instrumentation                                |

**Installation:**
```bash
uv pip install "nvidia-nat[crewai]"
```

### LangChain/LangGraph

LangChain is a framework for building applications that utilize large language models (LLMs) to interact with data. It provides a set of tools for creating chains of LLM calls, allowing for complex workflows powered by LLMs. LangChain focuses on modularity and extensibility, making it suitable for integrating custom data pipelines and enhancing intelligent applications.

For more information, visit the [LangChain website](https://www.langchain.com/).


| Capability              | Providers / Details                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **LLM Providers**       | NVIDIA NIM, OpenAI, Azure OpenAI, AWS Bedrock, LiteLLM                              |
| **Embedder Providers**  | NVIDIA NIM, OpenAI, Azure OpenAI                                                    |
| **Retriever Providers** | NeMo Retriever, Milvus                                                              |
| **Tool Calling**        | Fully supported through LangChain's `StructuredTool` interface                      |
| **Profiling**           | Comprehensive profiling support with callback handlers                              |


**Installation:**
```bash
uv pip install "nvidia-nat[langchain]"
```

### LlamaIndex

LlamaIndex is a powerful framework for building applications that utilize large language models (LLMs) to query and interact with structured and unstructured data. It provides a set of tools for creating indexes over data sources—such as documents, databases, and APIs—enabling complex retrieval, question answering, and orchestration workflows powered by LLMs. LlamaIndex focuses on modularity and extensibility, making it suitable for integrating custom data pipelines and enhancing intelligent applications.

For more information, visit the [LlamaIndex website](https://www.llamaindex.ai/).

| Capability              | Providers / Details                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **LLM Providers**       | NVIDIA NIM, OpenAI, Azure OpenAI, AWS Bedrock, LiteLLM                              |
| **Embedder Providers**  | NVIDIA NIM, OpenAI, Azure OpenAI                                                    |
| **Retriever Providers** | None (Use LlamaIndex native retrievers)                                             |
| **Tool Calling**        | Fully supported through LlamaIndex's `FunctionTool` interface                       |
| **Profiling**           | Comprehensive profiling support with callback handlers                              |

**Installation:**
```bash
uv pip install "nvidia-nat[llama-index]"
```

### Semantic Kernel

Microsoft's Semantic Kernel is a framework for building applications that utilize large language models (LLMs) to interact with data. It provides a set of tools for creating kernels that can be used to create complex workflows powered by LLMs. Semantic Kernel focuses on modularity and extensibility, making it suitable for integrating custom data pipelines and enhancing intelligent applications.

For more information, visit the [Semantic Kernel website](https://learn.microsoft.com/en-us/semantic-kernel/).

| Capability              | Providers / Details                                                                 |
|-------------------------|-------------------------------------------------------------------------------------|
| **LLM Providers**       | OpenAI, Azure OpenAI                                                                |
| **Embedder Providers**  | None (use framework-agnostic embedders if needed)                                   |
| **Retriever Providers** | None (use Semantic Kernel native connectors)                                        |
| **Tool Calling**        | Fully supported through Semantic Kernel's function calling                          |
| **Profiling**           | Comprehensive profiling support with instrumentation                                |

**Installation:**
```bash
uv pip install "nvidia-nat[semantic-kernel]"
```
