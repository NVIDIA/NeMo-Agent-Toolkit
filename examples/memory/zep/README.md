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

# Zep Cloud Examples

These examples use the Zep Cloud memory backend.

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation and Setup](#installation-and-setup)
- [Run the Workflow](#run-the-workflow)
  - [Create Memory](#create-memory)
  - [Recall Memory](#recall-memory)

## Key Features

- **Zep Cloud v3 Memory Backend Integration:** Demonstrates how to integrate Zep Cloud v3 as a memory backend for NeMo Agent toolkit workflows, using the new thread-based API with automatic user and thread management.
- **Chat Memory Management:** Shows implementation of simple chat functionality with the ability to create, store, and recall memories using Zep Cloud's thread and context APIs.
- **Advanced Memory Features:** Leverages Zep Cloud v3's advanced features including automatic fact extraction, knowledge graph construction, and fast context retrieval (P95 < 200ms with basic mode).
- **Asynchronous Knowledge Graph:** Zep v3 builds a knowledge graph asynchronously in the background, extracting facts, entities, and relationships from conversations.

## Prerequisites

- Zep Cloud Account: Sign up for a Zep Cloud account at [Zep Cloud](https://www.getzep.com/)
- Zep API Key: Obtain your API key from the Zep Cloud dashboard

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

To run this example, install the required dependencies by running the following command:
```bash
uv sync --extra langchain --extra zep-cloud --extra telemetry
```

### Configure Zep API Key

Set your Zep Cloud API key as an environment variable:

```bash
export ZEP_API_KEY=<your_zep_api_key>
```

Optionally, add this to your shell profile for persistence:
```bash
# For zsh (macOS default)
echo 'export ZEP_API_KEY=<your_zep_api_key>' >> ~/.zshrc

# For bash
echo 'export ZEP_API_KEY=<your_zep_api_key>' >> ~/.bashrc
```

### Start Observability (Optional)

The examples are configured to use the Phoenix observability tool. Start phoenix on `localhost:6006` with:

```bash
docker compose -f examples/deploy/docker-compose.phoenix.yml up
```

## Run the Workflow

This example demonstrates two approaches to memory management with Zep Cloud v3:

1. **Automatic Memory** (`react_agent_config.yml`) - Automatic capture and retrieval
2. **Hybrid Memory** (`react_agent_with_memory_tools_config.yml`) - Automatic capture with tool-based retrieval

Both use the `auto_memory_agent` wrapper which guarantees memory capture without requiring the LLM to remember to save memories.

Zep Cloud v3 automatically extracts facts and memories from conversations, builds a knowledge graph, and retrieves relevant context for future interactions.

**Important**: Zep v3 processes data asynchronously to build the knowledge graph. There may be a delay (typically a few seconds to a minute) between adding messages and having them appear in retrieved context. This is expected behavior as Zep extracts facts and builds relationships in the background.

### Approach 1: Automatic Memory (Recommended)

Uses `auto_memory_agent` with automatic retrieval enabled. Memory is automatically captured and injected before every agent response.

```bash
nat run --config_file=examples/memory/zep/configs/react_agent_config.yml --input "my favorite flavor is strawberry"
```

The agent automatically stores this fact to Zep's knowledge graph. Later:

```bash
nat run --config_file=examples/memory/zep/configs/react_agent_config.yml --input "what flavor of ice-cream should I get?"
```

**Expected Output:**
```console
Workflow Result:
['You should get strawberry ice cream, as it is your favorite flavor.']
```

The agent automatically retrieves relevant context from memory and uses it to personalize the response.

### Approach 2: Hybrid Memory (Tool-Based Retrieval)

Uses `auto_memory_agent` with automatic saving but tool-based retrieval. The agent decides when to call the `get_memory` tool.

```bash
nat run --config_file=examples/memory/zep/configs/react_agent_with_memory_tools_config.yml --input "my favorite flavor is strawberry"
```

Memory is automatically captured. When the agent needs context:

```bash
nat run --config_file=examples/memory/zep/configs/react_agent_with_memory_tools_config.yml --input "what flavor of ice-cream should I get?"
```

The agent will call the `get_memory` tool to retrieve relevant context before responding.

## Additional Features

Zep Cloud v3 provides additional capabilities that can be explored:

- **Thread-based Memory:** Organize memories by thread for different conversation contexts. This example uses `conversation_id` from NAT's Context as `thread_id`, enabling separate memory contexts per UI conversation.
- **Fact Extraction:** Automatically extract and store factual information from conversations in a knowledge graph
- **Graph-based Memory:** Use Zep's temporal knowledge graph capabilities for complex relationship tracking
- **Fast Context Retrieval:** Get assembled context with `mode="basic"` for sub-200ms retrieval (P95)
- **Customizable Context:** Use `mode="summary"` for LLM-summarized context blocks

### Context Management

NAT's Context object provides user identity and conversation tracking:
- `user_id`: Identifies the user (defaults to `"default_NAT_user"` if not provided)
- `conversation_id`: Maps to Zep's `thread_id` for multi-thread support
- `user_first_name`, `user_last_name`, `user_email`: Used for Zep user creation with fallbacks

If these fields are not provided in the HTTP request Context, warnings will be logged but execution will continue with sensible defaults.

## Zep v3 API Migration

This example uses Zep Cloud v3 API with the following key integrations:

### Zep v3 API Calls
- `user.add()`: Create users with profile information (done automatically by `ZepEditor`)
- `thread.create()`: Create threads for organizing conversations (done automatically per conversation)
- `thread.add_messages()`: Add chat messages to threads (called by `add_items()`)
- `thread.get_user_context(mode="basic"|"summary")`: Retrieve pre-formatted context (called by `retrieve_memory()`)

### NAT Memory Interface Implementation

The `ZepEditor` implements NAT's `MemoryEditor` interface:

- `add_items(items, user_id, **kwargs)`: Adds messages to Zep threads
  - Maps `conversation_id` from Context to `thread_id`
  - Supports `ignore_roles` kwarg for selective graph memory

- `retrieve_memory(query, user_id, **kwargs)`: Retrieves formatted context
  - Returns Zep's pre-formatted context string optimized for LLM consumption
  - Supports `mode` kwarg: `"basic"` (fast, P95 < 200ms) or `"summary"` (LLM-summarized)
  - Replaces v2's `memory.search_sessions` which returned unformatted results

- `remove_items(user_id, **kwargs)`: Removes user data (implementation-specific)

For more information about Zep Cloud v3 features and migration from v2, visit the [Zep Documentation](https://help.getzep.com/).

