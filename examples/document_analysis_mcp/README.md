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