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

# OpenAI Chat Completions API Compatible Endpoint

The NeMo Agent Toolkit provides full OpenAI Chat Completions API compatibility through a dedicated mode that enables seamless integration with existing OpenAI-compatible client libraries and workflows.

## Overview

When OpenAI compatible mode is enabled, the toolkit creates a single endpoint that fully implements the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) specification. This endpoint handles both streaming and non-streaming requests based on the `stream` parameter, exactly like the official OpenAI API.

### Key Benefits

- **Drop-in Replacement**: Works with existing OpenAI client libraries without code changes
- **Full API Compatibility**: Supports all OpenAI Chat Completions API parameters
- **Industry Standard**: Familiar interface for developers already using OpenAI
- **Future-Proof**: Aligned with established API patterns and ecosystem tools

## Configuration

To enable OpenAI compatible mode, set `openai_api_compatible: true` in your FastAPI front-end configuration:

```yaml
general:
  front_end:
    _type: fastapi
    workflow:
      method: POST
      openai_api_path: /v1/chat/completions
      openai_api_compatible: true  # Enable OpenAI compatible mode
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_compatible` | boolean | `false` | Enable OpenAI compatible mode |
| `openai_api_path` | string | `/chat` | Base path for the OpenAI endpoint |
| `method` | string | `POST` | HTTP method for the endpoint |

## Endpoint Behavior

### OpenAI Compatible Mode (`openai_api_compatible: true`)

Creates a single endpoint that handles both streaming and non-streaming requests:

- **Route**: `/v1/chat/completions` (configurable via `openai_api_path`)
- **Method**: POST
- **Content-Type**: `application/json`
- **Behavior**: Routes to streaming or non-streaming based on `stream` parameter

### Legacy Mode (`openai_api_compatible: false`)

Creates separate endpoints for different request types:

- **Non-streaming**: `/<openai_api_path>`
- **Streaming**: `<openai_api_path>/stream`

## API Specification

### Request Format

The endpoint accepts all standard OpenAI Chat Completions API parameters:

| Parameter | Type | Description | Validation |
|-----------|------|-------------|------------|
| `messages` | array | **Required.** List of messages in conversation format | min 1 item |
| `model` | string | Model identifier | - |
| `frequency_penalty` | number | Decreases likelihood of repeating tokens | -2.0 to 2.0 |
| `logit_bias` | object | Modify likelihood of specific tokens | token ID → bias |
| `logprobs` | boolean | Return log probabilities | - |
| `top_logprobs` | integer | Number of most likely tokens to return | 0 to 20 |
| `max_tokens` | integer | Maximum tokens to generate | ≥ 1 |
| `n` | integer | Number of completions to generate | 1 to 128 |
| `presence_penalty` | number | Increases likelihood of new topics | -2.0 to 2.0 |
| `response_format` | object | Specify response format | - |
| `seed` | integer | Random seed for deterministic outputs | - |
| `service_tier` | string | Service tier selection | "auto" or "default" |
| `stop` | string/array | Stop sequences | - |
| `stream` | boolean | Enable streaming responses | default: false |
| `stream_options` | object | Streaming configuration options | - |
| `temperature` | number | Sampling temperature | 0.0 to 2.0 |
| `top_p` | number | Nucleus sampling parameter | 0.0 to 1.0 |
| `tools` | array | Available function tools | - |
| `tool_choice` | string/object | Tool selection strategy | - |
| `parallel_tool_calls` | boolean | Enable parallel tool execution | default: true |
| `user` | string | End-user identifier | - |

### Response Format

#### Non-Streaming Response

```json
{
  "id": "chatcmpl-123456789",
  "object": "chat.completion",
  "created": 1704729600,
  "model": "nvidia/llama-3.1-8b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm an AI assistant ready to help you."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 12,
    "total_tokens": 22
  },
  "system_fingerprint": null,
  "service_tier": null
}
```

#### Streaming Response

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704729600,"model":"nvidia/llama-3.1-8b-instruct","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704729600,"model":"nvidia/llama-3.1-8b-instruct","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704729600,"model":"nvidia/llama-3.1-8b-instruct","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1704729600,"model":"nvidia/llama-3.1-8b-instruct","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":12,"total_tokens":22}}

data: [DONE]
```

## Usage Examples

### cURL Examples

#### Non-Streaming Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false,
    "temperature": 0.7,
    "max_tokens": 100
  }'
```

#### Streaming Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/llama-3.1-8b-instruct",
    "messages": [
      {"role": "user", "content": "Tell me a short story"}
    ],
    "stream": true,
    "temperature": 0.7
  }'
```

### Client Library Examples

#### OpenAI Python Client

```python
from openai import OpenAI

# Initialize client pointing to your NeMo Agent Toolkit server
client = OpenAI(
    api_key="not-needed",  # API key not required for local deployment
    base_url="http://localhost:8000/v1"
)

# Non-streaming chat completion
response = client.chat.completions.create(
    model="nvidia/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    stream=False,
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)

# Streaming chat completion
stream = client.chat.completions.create(
    model="nvidia/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "Write a haiku about technology"}
    ],
    stream=True,
    temperature=0.8
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

#### AI SDK (JavaScript/TypeScript)

```typescript
import { openai } from '@ai-sdk/openai';
import { generateText, streamText } from 'ai';

// Configure custom OpenAI provider
const customOpenAI = openai({
  baseURL: 'http://localhost:8000/v1',
  apiKey: 'not-needed'
});

// Non-streaming generation
const { text } = await generateText({
  model: customOpenAI('nvidia/llama-3.1-8b-instruct'),
  prompt: 'Explain the benefits of renewable energy',
  temperature: 0.7,
  maxTokens: 200
});

console.log(text);

// Streaming generation
const { textStream } = await streamText({
  model: customOpenAI('nvidia/llama-3.1-8b-instruct'),
  prompt: 'Describe the future of artificial intelligence',
  temperature: 0.7
});

for await (const textPart of textStream) {
  process.stdout.write(textPart);
}
```

#### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Initialize LangChain with custom base URL
llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="nvidia/llama-3.1-8b-instruct",
    temperature=0.7
)

# Use with LangChain
messages = [HumanMessage(content="What are the key principles of machine learning?")]
response = llm.invoke(messages)
print(response.content)

# Streaming with LangChain
for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)
```

## Advanced Features

### Function Calling

The endpoint supports OpenAI-style function calling when your workflow includes tool capabilities:

```python
response = client.chat.completions.create(
    model="nvidia/llama-3.1-8b-instruct",
    messages=[
        {"role": "user", "content": "What's the weather like in San Francisco?"}
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ],
    tool_choice="auto"
)
```

### Conversation History

Maintain conversation context by including previous messages:

```python
messages = [
    {"role": "user", "content": "What is machine learning?"},
    {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
    {"role": "user", "content": "Can you give me a practical example?"}
]

response = client.chat.completions.create(
    model="nvidia/llama-3.1-8b-instruct",
    messages=messages,
    temperature=0.7
)
```

## Migration Guide

### From Legacy Mode

If you're currently using legacy mode with separate endpoints:

1. **Update Configuration**: Set `openai_api_compatible: true`
2. **Update Client Code**: Use single endpoint with `stream` parameter
3. **Test Thoroughly**: Verify both streaming and non-streaming functionality
4. **Deploy Gradually**: Consider blue-green deployment for production

### From OpenAI API

If you're migrating from OpenAI's API:

1. **Update Base URL**: Point to your NeMo Agent Toolkit server
2. **Update Model Names**: Use your configured model identifiers
3. **Test Compatibility**: Verify all features work as expected
4. **Monitor Performance**: Check latency and throughput

## Error Handling

The endpoint returns standard HTTP status codes and OpenAI-compatible error responses:

```json
{
  "error": {
    "message": "Invalid request: temperature must be between 0.0 and 2.0",
    "type": "invalid_request_error",
    "param": "temperature",
    "code": "invalid_parameter"
  }
}
```

Common error scenarios:
- **400 Bad Request**: Invalid parameters or malformed request
- **422 Unprocessable Entity**: Request validation failed
- **500 Internal Server Error**: Server-side processing error

## Performance Considerations

### Streaming vs Non-Streaming

- **Use Streaming**: For real-time applications, chatbots, or when immediate feedback is needed
- **Use Non-Streaming**: For batch processing, when you need the complete response before proceeding

### Connection Management

For high-throughput applications:
- Use connection pooling in your HTTP client
- Configure appropriate timeout values
- Consider implementing retry logic with exponential back-off

## Monitoring and Observability

The OpenAI compatible endpoint integrates with the toolkit existing monitoring capabilities:

- **Request Metrics**: Track request volume, latency, and error rates
- **Model Performance**: Monitor token usage and generation speed
- **Resource Utilization**: CPU, memory, and GPU usage tracking

## Security Considerations

### Authentication

While the endpoint accepts an `api_key` parameter for OpenAI compatibility, authentication is typically handled at the infrastructure level:

- Use reverse proxy (nginx, Apache) for API key validation
- Implement rate limiting to prevent abuse
- Consider IP allow-listing for production deployments

### Data Privacy

- Requests and responses are not logged by default
- Implement audit logging if required for compliance
- Ensure secure transmission using HTTPS in production

## Troubleshooting

### Common Issues

1. **Connection Refused**: Verify server is running and accessible
2. **Invalid Model**: Check model name matches your configuration
3. **Timeout Errors**: Increase timeout values for complex requests
4. **Memory Issues**: Monitor resource usage for large context windows

### Debug Mode

Enable debug logging to troubleshoot issues:

```yaml
logging:
  level: DEBUG
  handlers:
    - console
```

## Related Documentation

- [NeMo Agent Toolkit Configuration Guide](../workflows/workflow-configuration.md)
- [API Server Endpoints](./api-server-endpoints.md)
- [WebSocket Messaging Interface](./websockets.md)
- [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat)