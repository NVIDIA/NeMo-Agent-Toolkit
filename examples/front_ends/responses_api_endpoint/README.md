<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# OpenAI Responses API Endpoint

**Complexity:** 🟢 Beginner

This example demonstrates how to configure a NeMo Agent toolkit FastAPI frontend to accept requests in the [OpenAI Responses API format](https://platform.openai.com/docs/api-reference/responses).

## Overview

The OpenAI Responses API uses a different request format than the Chat Completions API:

| Feature | Chat Completions API | Responses API |
|---------|---------------------|---------------|
| Input field | `messages` (array) | `input` (string or array) |
| System prompt | In messages array | `instructions` field |
| Response object | `chat.completion` | `response` |
| Streaming events | `chat.completion.chunk` | `response.created`, `response.output_text.delta`, etc. |

This example configures the `/v1/responses` endpoint to accept the Responses API format while the standard `/generate` and `/chat` endpoints continue using Chat Completions format.

> **⚠️ Important**: The Responses API format is provided for pass-through compatibility with managed services that support stateful backends (such as OpenAI and Azure OpenAI). NeMo Agent toolkit workflows do not inherently support stateful backends. Features like `previous_response_id` will be accepted but ignored.

## Prerequisites

1. **Install LangChain integration** (required for `tool_calling_agent` workflow):

```bash
uv pip install -e '.[langchain]'
```

2. **Set up the NVIDIA API key**:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Start the Server

```bash
nat serve --config_file examples/front_ends/responses_api_endpoint/configs/config.yml
```

The server will start on port 8088 with the following endpoints:

| Endpoint | Format | Description |
|----------|--------|-------------|
| `/generate` | NAT default | Standard workflow endpoint |
| `/chat` | Chat Completions | OpenAI Chat Completions format |
| `/chat/stream` | Chat Completions | Streaming Chat Completions |
| `/v1/responses` | Responses API | OpenAI Responses API format |

## Test with curl

### Responses API Format (Non-Streaming)

```bash
curl -X POST http://localhost:8088/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "input": "What time is it?"
  }'
```

**Expected Response:**

```json
{
  "id": "resp_abc123...",
  "object": "response",
  "status": "completed",
  "model": "gpt-4o-mini",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The current time is..."
        }
      ]
    }
  ]
}
```

### Responses API Format (Streaming)

```bash
curl -X POST http://localhost:8088/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "input": "What time is it?",
    "stream": true
  }'
```

**Expected SSE Events:**

```
event: response.created
data: {"type": "response.created", "response": {"id": "resp_...", "status": "in_progress"}}

event: response.output_item.added
data: {"type": "response.output_item.added", ...}

event: response.output_text.delta
data: {"type": "response.output_text.delta", "delta": "The current"}

event: response.output_text.delta
data: {"type": "response.output_text.delta", "delta": " time is..."}

event: response.done
data: {"type": "response.done", "response": {"status": "completed", ...}}
```

### With System Instructions

```bash
curl -X POST http://localhost:8088/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "input": "What time is it?",
    "instructions": "You are a helpful assistant. Always be concise."
  }'
```

### With Tools

```bash
curl -X POST http://localhost:8088/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gpt-4o-mini",
    "input": "What time is it?",
    "tools": [
      {
        "type": "function",
        "name": "current_datetime",
        "description": "Get the current date and time"
      }
    ]
  }'
```

### Chat Completions Format (Still Works)

The `/chat` endpoint continues to use the Chat Completions format:

```bash
curl -X POST http://localhost:8088/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "What time is it?"}]
  }'
```

## Configuration Options

### Using Explicit Format Override

If you want to use the Responses API format on a custom path (not containing "responses"), use the explicit `openai_api_v1_format` setting:

```yaml
general:
  front_end:
    _type: fastapi
    workflow:
      openai_api_v1_path: /v1/custom/endpoint
      openai_api_v1_format: responses  # Force Responses API format
```

Available format options:
- `auto` (default): Detects based on path pattern
- `chat_completions`: Force Chat Completions API format
- `responses`: Force Responses API format

## Limitations

- **No Stateful Backend**: `previous_response_id` is accepted but ignored
- **No Built-in Tools**: OpenAI built-in tools like `code_interpreter` are not executed by NAT; use the `responses_api_agent` workflow type for that functionality
- **Tool Format Conversion**: Responses API tool definitions are converted to Chat Completions format internally

## Related Examples

- [Tool Calling Agent with Responses API](../../agents/tool_calling/README.md#using-tool-calling-with-the-openai-responses-api) - For using OpenAI's Responses API directly with built-in tools
- [Simple Auth](../simple_auth/README.md) - Authentication example
- [Custom Routes](../simple_calculator_custom_routes/README.md) - Custom endpoint routes

