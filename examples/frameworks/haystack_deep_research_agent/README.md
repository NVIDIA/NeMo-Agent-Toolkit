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

1. **Web Search Tool**: Uses Haystack's SerperDevWebSearch and LinkContentFetcher to search the web and extract content from web pages
2. **RAG Tool**: Uses OpenSearchDocumentStore to index and query internal documents with semantic retrieval
3. **Deep Research Agent** (`register.py`): Orchestrates the agent and imports modular pipelines from `src/nat_haystack_deep_research_agent/pipelines/`:
   - `search.py`: builds the web search tool
   - `rag.py`: builds the RAG pipeline and tool
   - `indexing.py`: startup indexing (PDF/TXT/MD) into OpenSearch

## Prerequisites

Before using this workflow, ensure you have:

1. **NVIDIA API Key**: Required for the chat generator and RAG functionality
   - Get your key from [NVIDIA API Catalog](https://build.nvidia.com/)
   - Set as environment variable: `export NVIDIA_API_KEY=your_key_here`

2. **SerperDev API Key**: Required for web search functionality
   - Get your key from [SerperDev](https://serper.dev)
   - Set as environment variable: `export SERPERDEV_API_KEY=your_key_here`

3. **OpenSearch Instance**: Required for RAG functionality
   - You can run OpenSearch locally using `docker`

## Installation and Usage

Follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NVIDIA NeMo Agent Toolkit.

### Step 1: Set Your API Keys

```bash
export NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>
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
uv pip install -e examples/frameworks/haystack_deep_research_agent
```

### Step 4: Add Sample Documents (Optional)

Place documents in the example `data/` directory to enable RAG (PDF, TXT, or MD). On startup, the workflow indexes files from:

- `workflow.data_dir` (default: `/data`)
- If empty/missing, it falls back to this example's bundled `data/` directory

```bash
# Example: Download a sample PDF
wget "https://docs.aws.amazon.com/pdfs/bedrock/latest/userguide/bedrock-ug.pdf" \
  -O examples/frameworks/haystack_deep_research_agent/data/bedrock-ug.pdf
```

### Step 5: Run the Workflow

```bash
nat run --config_file=examples/frameworks/haystack_deep_research_agent/configs/config.yml --input "What are the latest updates on the Artemis moon mission?"
```

## Example Queries

Here are some example queries you can try:

**Web Search Examples:**

```bash
# Current events
nat run --config_file=examples/frameworks/haystack_deep_research_agent/configs/config.yml --input "What are the latest developments in AI research for 2024?"

# Technology news
nat run --config_file=examples/frameworks/haystack_deep_research_agent/configs/config.yml --input "What are the new features in the latest Python release?"
```

**RAG Examples (if you have documents indexed):**

```bash
# Document-specific queries
nat run --config_file=examples/frameworks/haystack_deep_research_agent/configs/config.yml --input "What are the key features of AWS Bedrock?"

# Mixed queries (will use both web search and RAG)
nat run --config_file=examples/frameworks/haystack_deep_research_agent/configs/config.yml --input "How does AWS Bedrock compare to other AI platforms in 2024?"
```

**Web Search + RAG Examples:**

```bash
nat run --config_file=examples/frameworks/haystack_deep_research_agent/configs/config.yml --input "Is panna (heavy cream) needed on carbonara? Check online the recipe and compare it with the one from our internal dataset."
```

## Testing

### Quick smoke test (no external services)

- Validates the workflow config without hitting LLMs or OpenSearch.

```bash
# In your virtual environment
pytest -q examples/frameworks/haystack_deep_research_agent/tests -k config_yaml_loads_and_has_keys
```

### End-to-end test (requires keys + OpenSearch)

- Prerequisites:
  - Set keys: `NVIDIA_API_KEY` and `SERPERDEV_API_KEY`
  - OpenSearch running on `http://localhost:9200` (start with Docker):

```bash
docker run -d --name opensearch -p 9200:9200 -p 9600:9600 \
  -e "discovery.type=single-node" \
  -e "plugins.security.disabled=true" \
  opensearchproject/opensearch:2.11.1
```

- Run the e2e test (ensure `pytest-asyncio` is installed in your virtual environment):

```bash
pip install pytest-asyncio  # if not already installed
export NVIDIA_API_KEY=<YOUR_KEY>
export SERPERDEV_API_KEY=<YOUR_KEY>
pytest -q examples/frameworks/haystack_deep_research_agent/tests -k full_workflow_e2e
```

## Configuration

The workflow is configured via `config.yml`. Key configuration options include:

- **Web Search Tool**:
  - `top_k`: Number of search results to retrieve (default: 10)
  - `timeout`: Timeout for fetching web content (default: 3 seconds)
  - `retry_attempts`: Number of retry attempts for failed requests (default: 2)

- **RAG Tool**:
  - `opensearch_url`: OpenSearch host URL (default: `http://localhost:9200`)
  - `index_name`: OpenSearch index name (fixed: `deep_research_docs`)
  - `top_k`: Number of documents to retrieve (default: 15)
  - `index_on_startup`: If true, run indexing pipeline on start
  - `data_dir`: Directory to scan for documents; if empty/missing, falls back to example `data/`

- **Agent**:
  - `max_agent_steps`: Maximum number of agent steps (default: 20)
  - `system_prompt`: Customizable system prompt for the agent

## Customization

You can customize the workflow by:

1. **Modifying the system prompt** in `config.yml` to change the agent's behavior
2. **Adding more document types** by extending the RAG tool to support other file formats
3. **Changing the LLM model** by updating the top-level `llms` section in `config.yml`. This example defines `agent_llm` and `rag_llm` using the `nim` provider so they can leverage common parameters like `temperature`, `top_p`, and `max_tokens`. The workflow references them via the builder. See Haystack's NvidiaChatGenerator docs: [NvidiaChatGenerator](https://docs.haystack.deepset.ai/docs/nvidiachatgenerator)
4. **Adjusting search parameters** to optimize for your use case

## Troubleshooting

**Common Issues:**

1. **OpenSearch Connection Error**: Ensure OpenSearch is running and accessible at the configured host
2. **Missing API Keys**: Verify that both NVIDIA_API_KEY and SERPERDEV_API_KEY are set
3. **No Documents Found**: Check that PDF files are placed in the data directory and the path is correct
4. **Web Search Fails**: Verify your SerperDev API key is valid and has remaining quota

**Logs**: Check the NeMo-Agent-Toolkit logs for detailed error information and debugging.

## Architecture Details

The workflow demonstrates several key NeMo-Agent-Toolkit patterns:

- **Workflow Registration**: The agent is exposed as a workflow function with a Pydantic config
- **Builder LLM Integration**: LLMs are defined under top-level `llms:` and accessed via `builder.get_llm_config(...)`
- **Component Integration**: Haystack components are composed into tools within the workflow
- **Error Handling**: Robust error handling with fallback behaviors
- **Async Operations**: All operations are asynchronous for better performance

This example showcases how the Haystack AI framework can be seamlessly integrated into NeMo-Agent-Toolkit workflows while maintaining the flexibility and power of the underlying architecture.
