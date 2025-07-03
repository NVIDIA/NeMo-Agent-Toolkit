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

# Simple Calculator - Custom Routes and Metadata Access

This example demonstrates **custom API routes and request metadata access** using the Simple Calculator workflow. Learn how to extend AIQ toolkit applications with custom endpoints and access rich HTTP request context.

## 🎯 What You'll Learn

- **Custom API Routes**: Define and register custom endpoints dynamically
- **Request Metadata Access**: Capture HTTP headers, query parameters, and more
- **API Extension**: Extend AIQ toolkit applications with specialized endpoints
- **Context Management**: Access request context within your functions
- **Production APIs**: Build sophisticated API interfaces for AI workflows

## 🔗 Prerequisites

This example builds upon the [basic Simple Calculator](../../../basic/functions/simple_calculator/). Install it first:

```bash
uv pip install -e examples/basic/functions/simple_calculator
```

## 📦 Installation

```bash
uv pip install -e examples/intermediate/custom_routes/simple_calculator
```

## 🚀 Usage

### Start the API Server

```bash
aiq serve --config_file examples/intermediate/custom_routes/simple_calculator/configs/config-metadata.yml
```

The server will start with both default and custom endpoints:
- Standard endpoint: `POST /generate`
- Custom endpoint: `POST /get_request_metadata`

### Test Custom Routes

#### Access Request Metadata
```bash
curl -X 'POST' \
  'http://localhost:8000/get_request_metadata' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'X-Custom-Header: test-value' \
  -d '{"input_message": "show me request details"}'
```

#### Standard Calculator with Metadata Context
```bash
curl -X 'POST' \
  'http://localhost:8000/generate?user_id=123&session=abc' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d '{"input_message": "What is 5 + 3?"}'
```

## 🔍 Key Features Demonstrated

- **Dynamic Route Registration**: Add custom endpoints via configuration
- **Rich Metadata Access**: Capture comprehensive request information
- **Context Propagation**: Request context available throughout execution
- **HTTP Integration**: Full HTTP protocol support
- **Custom Logic**: Implement specialized business logic in custom routes

## 📊 Available Request Metadata

The custom endpoint captures:

| Metadata | Description | Example |
|----------|-------------|---------|
| `method` | HTTP method | `POST`, `GET` |
| `url_path` | Request path | `/get_request_metadata` |
| `url_scheme` | Protocol scheme | `http`, `https` |
| `headers` | HTTP headers | `Authorization`, `Content-Type` |
| `query_params` | URL parameters | `?user_id=123&session=abc` |
| `path_params` | Path variables | `/users/{user_id}` |
| `client_host` | Client IP address | `192.168.1.100` |
| `client_port` | Client port | `12345` |
| `cookies` | HTTP cookies | Session, authentication cookies |

## ⚙️ Configuration

The `config-metadata.yml` demonstrates:

```yaml
front_end:
  _type: fastapi
  endpoints:
    - path: /get_request_metadata
      method: POST
      description: Gets the request attributes from the request.
      function_name: current_request_attributes
```

## 🛠️ Custom Function Implementation

The example shows how to access metadata in your functions:

```python
async def custom_function(input: str) -> str:
    from aiq.builder.context import AIQContext
    context = AIQContext.get()

    # Access request metadata
    headers = context.metadata.headers
    query_params = context.metadata.query_params
    client_host = context.metadata.client_host

    # Use metadata in your logic
    return f"Request from {client_host}"
```

## 🌟 Use Cases

- **Authentication & Authorization**: Access tokens and user context
- **Request Routing**: Route based on headers or parameters
- **Audit Logging**: Track requests with full context
- **Rate Limiting**: Implement per-client rate limiting
- **Personalization**: Customize responses based on request metadata
- **Analytics**: Collect detailed usage analytics

## 🔧 Advanced Features

- **Custom processing**: Add custom processing for pre/post request handling
- **Response Customization**: Modify responses based on request context
- **Error Handling**: Context-aware error responses
- **Session Management**: Stateful interactions using request metadata

This example showcases how AIQ toolkit enables building sophisticated, production-ready APIs with rich request context and custom business logic.
