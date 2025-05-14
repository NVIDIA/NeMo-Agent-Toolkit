<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Document Analysis MCP Example

This example demonstrates how to use AIQ Toolkit with Model Context Protocol (MCP) to create a document analysis and question answering system. It showcases the integration of multiple tools and sophisticated configurations within the AIQ Toolkit framework.

## Features

- URL content fetching with HTML parsing
- Document analysis and information extraction
- Question answering about analyzed documents
- Enhanced error handling and retry mechanisms
- Docker support for easy deployment
- Full MCP server and client implementation

## Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose
- NVIDIA API Key for accessing the LLM
- AIQ Toolkit installed with required plugins

## Setup

1. Set your NVIDIA API Key:
   ```bash
   export NVIDIA_API_KEY=your_api_key_here
   ```

2. Install required AIQ Toolkit plugins:
   ```bash
   uv pip install -e '.[langchain]'
   ```

3. Build and start the Docker container:
   ```bash
   docker-compose -f deployment/docker-compose.yml up --build
   ```

4. The server will be available at `http://localhost:9902`

## Available Tools

1. **Fetch Tool**
   - Fetches content from a URL
   - Parses HTML and extracts text
   - Handles errors and timeouts
   ```python
   {
       "url": "https://example.com"
   }
   ```

2. **Document Analysis Tool**
   - Analyzes document text
   - Splits into chunks
   - Creates vector store for Q&A
   ```python
   {
       "text": "Your document text here"
   }
   ```

3. **Question Answering Tool**
   - Answers questions about analyzed documents
   - Uses vector search for context
   - Provides detailed answers
   ```python
   {
       "question": "Your question here"
   }
   ```

## Architecture

- `Dockerfile`: Container configuration
- `deployment/docker-compose.yml`: Service orchestration

## How it Works

1. The server provides three main tools:
   - URL content fetching
   - Document analysis
   - Question answering

2. Each tool has:
   - Input validation
   - Error handling
   - Retry mechanisms
   - Detailed logging

3. The system uses:
   - LangChain for document processing
   - FAISS for vector storage
   - BeautifulSoup for HTML parsing
   - Docker for deployment


## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install AIQ Toolkit.

To run this example do the following:

1. Start up docker compose using the provided `docker-compose.yml` file.
 ```bash
 docker compose -f examples/document_analysis_mcp/deployment/docker-compose.yml up -d
 ```
 The container will pull down the necessary code to run the server when it starts, so it may take a few minutes before the server is ready.
 You can inspect the logs by running
 ```bash
 docker compose -f examples/document_analysis_mcp/deployment/docker-compose.yml logs
 ```
 The server is ready when you see the following:
 ```bash
 mcp-proxy-aiq  | INFO:     Started server process [1]
 mcp-proxy-aiq  | INFO:     Waiting for application startup.
 mcp-proxy-aiq  | INFO:     Application startup complete.
 mcp-proxy-aiq  | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
 ```

 2.  In a new terminal, from the root of the AIQ Toolkit repository run the workflow:
 ```bash
 source .venv/bin/activate
 aiq run --config_file=examples/document_analysis_mcp/configs/config.yml --input="What is langchain?"
 ```

 The ReAct Agent will use the tool to answer the question
 ```console
 2025-03-11 16:13:29,922 - aiq.agent.react_agent.agent - INFO - The agent's thoughts are:
Thought: To answer this question, I need to find out what LangChain is. It's possible that it's a recent development or a concept that has been discussed online. I can use the internet to find the most up-to-date information about LangChain.

Action: mcp_url_tool
Action Input: {"url": "https://langchain.dev/", "max_length": 5000, "start_index": 0, "raw": false}


2025-03-11 16:13:29,924 - aiq.agent.react_agent.agent - INFO - Calling tool mcp_url_tool with input: {"url": "https://langchain.dev/", "max_length": 5000, "start_index": 0, "raw": false}
```
```console
Workflow Result:
["LangChain is a composable framework that supports developers in building, running, and managing applications powered by Large Language Models (LLMs). It offers a suite of products, including LangChain, LangGraph, and LangSmith, which provide tools for building context-aware and reasoning applications, deploying LLM applications at scale, and debugging, collaborating, testing, and monitoring LLM apps. LangChain's products are designed to help developers create reliable and efficient GenAI applications, and its platform is used by teams of all sizes across various industries."]
```

## Usage Examples

1. Fetch content from a URL:
   ```bash
   curl -X POST http://localhost:9902/tools/fetch \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'
   ```
2. Analyze a document:
   ```bash
   curl -X POST http://localhost:9902/tools/analyze_document \
     -H "Content-Type: application/json" \
     -d '{"text": "Your document text here"}'
   ```

3. Ask a question:
   ```bash
   curl -X POST http://localhost:9902/tools/answer_question \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic?"}'
   ```

## Related Documentation

- [AIQ Toolkit Documentation](https://docs.nvidia.com/aiqtoolkit)
- [MCP Server Guide](./docs/source/workflows/mcp/mcp-server.md)
- [MCP Client Guide](./docs/source/workflows/mcp/mcp-client.md)
- [LangChain Integration](./docs/source/plugins/langchain.md)
