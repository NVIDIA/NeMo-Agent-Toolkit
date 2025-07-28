<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS-IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Simple Calculator - Custom Routes and Metadata Access

This example demonstrates how to extend NVIDIA NeMo Agent toolkit applications with custom API routes and HTTP request metadata access. Build sophisticated APIs that capture rich request context for authentication, routing, and specialized business logic.

## Table of Contents

- [Key Features](#key-features)
- [What You'll Learn](#what-youll-learn)
- [Configuration][#configuration]
- [Installation and Setup](#installation-and-setup)
  - [Install this Workflow](#install-this-workflow)
  - [Set Up API Keys](#set-up-api-keys)
- [Example Usage](#example-usage)
  - [Run the Workflow](#run-the-workflow)

## Key Features

- **Custom API Route Registration:** Demonstrates how to define and register custom endpoints through YAML configuration that are dynamically added to the FastAPI server alongside standard Agent toolkit endpoints.
- **HTTP Request Metadata Access:** Shows comprehensive capture of HTTP request context including method, URL path, headers, query parameters, client information, and cookies through the `AIQContext` system.
- **Context Management Integration:** Uses the `aiq.builder.context.AIQContext.get()` method to access request metadata throughout function execution, enabling sophisticated request-aware business logic.
- **Production API Extension Patterns:** Provides patterns for building production-ready APIs with specialized endpoints for authentication, routing, and custom business logic while maintaining Agent toolkit workflow capabilities.
- **FastAPI Integration:** Demonstrates seamless integration with FastAPI framework features while leveraging Agent toolkit workflow execution and function registration system.

## What You'll Learn

- **Custom API routes**: Define and register custom endpoints through configuration
- **Request metadata access**: Capture HTTP headers, query parameters, and client information
- **Context management**: Access request context throughout function execution
- **API extension patterns**: Build production-ready APIs with specialized endpoints

## Configuration

Users can define custom routes that are dynamically added to the API server, and capture HTTP request metadata such as the method, URL path, URL scheme, headers, query parameters, path parameters, host, port, and cookies.

### Defining Custom Routes

Add custom endpoints in your configuration file's `front_end` section:

```yaml
general:
  use_uvloop: true

  front_end:
    _type: fastapi
    endpoints:
      - path: /get_request_metadata
        method: POST
        description: "Gets the request attributes from the request."
        function_name: current_request_attributes
```

### Complete Metadata Access Example
Get the instance of the `aiq.builder.context.AIQContext` object using the `aiq.builder.context.AIQContext.get()` method. This will give you access to the metadata method which holds the request attributes defined by the user on request. A complete example of the function can be found in `src/aiq/tool/server_tools.py`.

```python
@register_function(config_type=RequestAttributesTool)
async def current_request_attributes(config: RequestAttributesTool, builder: Builder):

    from starlette.datastructures import Headers
    from starlette.datastructures import QueryParams

    async def _get_request_attributes(unused: str) -> str:

        from aiq.builder.context import AIQContext
        aiq_context = AIQContext.get()

        method: str | None = aiq_context.metadata.method
        url_path: str | None = aiq_context.metadata.url_path
        url_scheme: str | None = aiq_context.metadata.url_scheme
        headers: Headers | None = aiq_context.metadata.headers
        query_params: QueryParams | None = aiq_context.metadata.query_params
        path_params: dict[str, str] | None = aiq_context.metadata.path_params
        client_host: str | None = aiq_context.metadata.client_host
        client_port: int | None = aiq_context.metadata.client_port
        cookies: dict[str, str] | None = aiq_context.metadata.cookies
        conversation_id: str | None = aiq_context.conversation_id

    yield FunctionInfo.from_fn(_get_request_attributes,
                               description="Returns the acquired user defined request attriubutes.")
```

## Installation and Setup

If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent toolkit.

### Install this Workflow:

From the root directory of the NeMo Agent toolkit library, run the following commands:

```bash
uv pip install -e examples/front_ends/simple_calculator_custom_routes
```

### Set Up API Keys
If you have not already done so, follow the [Obtaining API Keys](../../../docs/source/quick-start/installing.md#obtaining-api-keys) instructions to obtain an NVIDIA API key. You need to set your NVIDIA API key as an environment variable to access NVIDIA AI services:

```bash
export NVIDIA_API_KEY=<YOUR_API_KEY>
```

## Example Usage

### Run the Workflow

```bash
aiq serve --config_file examples/front_ends/simple_calculator_custom_routes/configs/config-metadata.yml
```

The server starts with both standard and custom endpoints:

- **Standard endpoint**: `POST /generate` - Default Agent toolkit workflow endpoint
- **Custom endpoint**: `POST /get_request_metadata` - Demonstrates metadata access



Access comprehensive request metadata:

```bash
curl -X 'POST' \
  'http://localhost:8000/get_request_metadata' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d '{"unused": "show me request details"}'
```
