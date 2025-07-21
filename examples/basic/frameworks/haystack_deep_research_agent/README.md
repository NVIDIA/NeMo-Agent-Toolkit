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

# Haystack Deep Research Agent

This example demonstrates how to build a deep research agent using Haystack framework  that combines web search and Retrieval Augmented Generation (RAG) capabilities using the NeMo-Agent-Toolkit.

## Overview

The Haystack Deep Research Agent is an intelligent research assistant that can:

- **Web Search**: Search the internet for current information using SerperDev API
- **Document Retrieval**: Query an internal document database using RAG with OpenSearch
- **Comprehensive Research**: Combine both sources to provide thorough, well-cited research reports
- **Intelligent Routing**: Automatically decide when to use web search vs. internal documents

## Architecture

The workflow consists of three main components:

1. **Web Search Tool** (`web_search_tool.py`): Uses Haystack's SerperDevWebSearch and LinkContentFetcher to search the web and extract content from web pages
2. **RAG Tool** (`rag_tool.py`): Uses OpenSearchDocumentStore to index and query internal documents with semantic retrieval
3. **Deep Research Agent** (`deep_research_agent.py`): Combines both tools using Haystack's Agent framework with OpenAI for intelligent orchestration

## Prerequisites

Before using this workflow, ensure you have:

1. **OpenAI API Key**: Required for the chat generator and RAG functionality
   - Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Set as environment variable: `export OPENAI_API_KEY=your_key_here`

2. **SerperDev API Key**: Required for web search functionality
   - Get your key from [SerperDev](https://serper.dev/api-key)
   - Set as environment variable: `export SERPERDEV_API_KEY=your_key_here`

3. **OpenSearch Instance**: Required for RAG functionality
   - You can run OpenSearch locally using Docker:
     ```bash
     docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
       -e "discovery.type=single-node" \
       -e "plugins.security.disabled=true" \
       opensearchproject/opensearch:2.11.1
     ```

## Installation and Usage

Follow the instructions in the [Install Guide](../../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ toolkit.

### Step 1: Set Your API Keys

```bash
export OPENAI_API_KEY=<YOUR_OPENAI_API_KEY>
export SERPERDEV_API_KEY=<YOUR_SERPERDEV_API_KEY>
```

### Step 2: Start OpenSearch (if not already running)

```bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "plugins.security.disabled=true" \
  opensearchproject/opensearch:2.11.1
```

### Step 3: Install the Workflow

```bash
uv pip install -e examples/basic/frameworks/haystack_deep_research_agent
```

### Step 4: Add Sample Documents (Optional)

Place PDF documents in the `data/` directory to enable RAG functionality:

```bash
# Example: Download a sample PDF
wget "https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf" \
  -O examples/basic/frameworks/haystack_deep_research_agent/data/bedrock-ug.pdf
```

### Step 5: Run the Workflow

```bash
aiq run --config_file=examples/basic/frameworks/haystack_deep_research_agent/src/aiq_haystack_deep_research_agent/configs/config.yml --input "What are the latest updates on the Artemis moon mission?"
```

## Example Queries

Here are some example queries you can try:

**Web Search Examples:**
```bash
# Current events
aiq run --config_file=examples/basic/frameworks/haystack_deep_research_agent/src/aiq_haystack_deep_research_agent/configs/config.yml --input "What are the latest developments in AI research for 2024?"

# Technology news
aiq run --config_file=examples/basic/frameworks/haystack_deep_research_agent/src/aiq_haystack_deep_research_agent/configs/config.yml --input "What are the new features in the latest Python release?"
```

**RAG Examples (if you have documents indexed):**
```bash
# Document-specific queries
aiq run --config_file=examples/basic/frameworks/haystack_deep_research_agent/src/aiq_haystack_deep_research_agent/configs/config.yml --input "What are the key features of AWS Bedrock?"

# Mixed queries (will use both web search and RAG)
aiq run --config_file=examples/basic/frameworks/haystack_deep_research_agent/src/aiq_haystack_deep_research_agent/configs/config.yml --input "How does AWS Bedrock compare to other AI platforms in 2024?"
```

## Configuration

The workflow is configured via `config.yml`. Key configuration options include:

- **Web Search Tool**:
  - `top_k`: Number of search results to retrieve (default: 10)
  - `timeout`: Timeout for fetching web content (default: 3 seconds)
  - `retry_attempts`: Number of retry attempts for failed requests (default: 2)

- **RAG Tool**:
  - `document_store_host`: OpenSearch host URL (default: "http://localhost:9200")
  - `index_name`: OpenSearch index name (default: "deep_research_docs")
  - `top_k`: Number of documents to retrieve (default: 15)
  - `data_directory`: Directory containing PDF documents to index

- **Agent**:
  - `max_agent_steps`: Maximum number of agent steps (default: 20)
  - `system_prompt`: Customizable system prompt for the agent

## Customization

You can customize the workflow by:

1. **Modifying the system prompt** in `config.yml` to change the agent's behavior
2. **Adding more document types** by extending the RAG tool to support other file formats
3. **Changing the LLM model** by updating the OpenAI model in the configuration
4. **Adjusting search parameters** to optimize for your use case

## Troubleshooting

**Common Issues:**

1. **OpenSearch Connection Error**: Ensure OpenSearch is running and accessible at the configured host
2. **Missing API Keys**: Verify that both OPENAI_API_KEY and SERPERDEV_API_KEY are set
3. **No Documents Found**: Check that PDF files are placed in the data directory and the path is correct
4. **Web Search Fails**: Verify your SerperDev API key is valid and has remaining quota

**Logs**: Check the NeMo-Agent-Toolkit logs for detailed error information and debugging.

## Architecture Details

The workflow demonstrates several key NeMo-Agent-Toolkit patterns:

- **Function Registration**: Each tool is registered as a function with its own configuration
- **Builder Pattern**: The NeMo-Agent-Toolkit Builder is used to create and manage tools and LLMs
- **Component Integration**: Haystack components are wrapped as NeMo-Agent-Toolkit functions
- **Error Handling**: Robust error handling with fallback behaviors
- **Async Operations**: All operations are asynchronous for better performance

This example showcases how the Haystack AI framework can be seamlessly integrated into NeMo-Agent-Toolkit workflows while maintaining the flexibility and power of the underlying architecture.
