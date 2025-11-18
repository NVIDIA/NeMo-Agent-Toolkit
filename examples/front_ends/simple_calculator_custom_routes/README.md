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

- [Simple Calculator - Custom Routes and Metadata Access](#simple-calculator---custom-routes-and-metadata-access)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [What You'll Learn](#what-youll-learn)
  - [Configuration](#configuration)
    - [Defining Custom Routes](#defining-custom-routes)
    - [Complete Metadata Access Example](#complete-metadata-access-example)
  - [Installation and Setup](#installation-and-setup)
    - [Install this Workflow:](#install-this-workflow)
    - [Set Up API Keys](#set-up-api-keys)
  - [Example Usage](#example-usage)
    - [Run the Workflow](#run-the-workflow)
    - [Additional Request Body Examples](#additional-request-body-examples)
      - [JSON Array](#json-array)
      - [JSON String](#json-string)
      - [JSON Number](#json-number)
      - [JSON Boolean](#json-boolean)

## Key Features

- **Custom API Route Registration:** Demonstrates how to define and register custom endpoints through YAML configuration that are dynamically added to the FastAPI server alongside standard Agent toolkit endpoints.
- **HTTP Request Metadata Access:** Shows comprehensive capture of HTTP request context including method, URL path, headers, query parameters, client information, and cookies through the `Context` system.
- **Context Management Integration:** Uses the `nat.builder.context.Context.get()` method to access request metadata throughout function execution, enabling sophisticated request-aware business logic.
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
  front_end:
    _type: fastapi
    endpoints:
      - path: /get_request_metadata
        method: POST
        description: "Gets the request attributes from the request."
        function_name: current_request_attributes
```

### Complete Metadata Access Example
Get the instance of the `nat.builder.context.Context` object using the `nat.builder.context.Context.get()` method. This will give you access to the metadata method which holds the request attributes defined by the user on request. A complete example of the function can be found in `src/nat/tool/server_tools.py`.

> [!NOTE]
>
> To accept arbitrary JSON payloads of any type (objects, arrays, strings, numbers, Boolean values) use Pydantic's `RootModel[JsonValue]`. This allows the function to receive any valid JSON type. Access the raw data through the `.root` attribute.
>
> Custom routes using `RootModel` do not support async generation (background jobs) as `RootModel` schemas are incompatible with the async generation field injection. Custom routes using `RootModel` are intended for direct request-response patterns.

```python
@register_function(config_type=RequestAttributesTool)
async def current_request_attributes(config: RequestAttributesTool, builder: Builder):

    from pydantic import RootModel
    from pydantic.types import JsonValue
    from starlette.datastructures import Headers, QueryParams

    class RequestBody(RootModel[JsonValue]):
        """
        Data model that accepts a request body of any valid JSON type.
        """
        root: JsonValue

    async def _get_request_attributes(request_body: RequestBody) -> str:

        from nat.builder.context import Context
        nat_context = Context.get()

        # Access request attributes from context
        method: str | None = nat_context.metadata.method
        url_path: str | None = nat_context.metadata.url_path
        url_scheme: str | None = nat_context.metadata.url_scheme
        headers: Headers | None = nat_context.metadata.headers
        query_params: QueryParams | None = nat_context.metadata.query_params
        path_params: dict[str, str] | None = nat_context.metadata.path_params
        client_host: str | None = nat_context.metadata.client_host
        client_port: int | None = nat_context.metadata.client_port
        cookies: dict[str, str] | None = nat_context.metadata.cookies
        conversation_id: str | None = nat_context.conversation_id

        # Access the request body data - can be any valid JSON type
        request_body_data: JsonValue = request_body.root

        return (f"Method: {method}, "
                f"URL Path: {url_path}, "
                f"URL Scheme: {url_scheme}, "
                f"Headers: {dict(headers) if headers is not None else 'None'}, "
                f"Query Params: {dict(query_params) if query_params is not None else 'None'}, "
                f"Path Params: {path_params}, "
                f"Client Host: {client_host}, "
                f"Client Port: {client_port}, "
                f"Cookies: {cookies}, "
                f"Conversation Id: {conversation_id}, "
                f"Request Body: {request_body_data}")

    yield FunctionInfo.from_fn(_get_request_attributes,
                               description="Returns the acquired user defined request attributes.")
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
nat serve --config_file examples/front_ends/simple_calculator_custom_routes/configs/config-metadata.yml
```

The server starts with both standard and custom endpoints:

- **Standard endpoint**: `POST /generate` - Default Agent toolkit workflow endpoint
- **Custom endpoint**: `POST /get_request_metadata` - Demonstrates metadata access

Access comprehensive request metadata:

```bash
curl -X POST http://localhost:8000/get_request_metadata \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d '{"message": "show me request details", "user_id": 123, "tags": ["test", "demo"], "active": true}'
```

Expected Response Format:

<!-- path-check-skip-begin -->
```console
{"value":"Method: POST, URL Path: /get_request_metadata, URL Scheme: http, Headers: {'host': 'localhost:8000', 'user-agent': 'curl/8.7.1', 'accept': 'application/json', 'content-type': 'application/json', 'authorization': 'Bearer token123', 'content-length': '95'}, Query Params: {}, Path Params: {}, Client Host: ::1, Client Port: 56922, Cookies: {}, Conversation Id: None, Request Body: {'message': 'show me request details', 'user_id': 123, 'tags': ['test', 'demo'], 'active': True}"}
```
<!-- path-check-skip-end -->

### Additional Request Body Examples

The following examples demonstrate the different JSON primitive types supported by the `RootModel[JsonValue]` implementation:

#### JSON Array

```bash
curl -X POST http://localhost:8000/get_request_metadata \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d '[1, 2, 3, 4, 5]'
```

#### JSON String

```bash
curl -X POST http://localhost:8000/get_request_metadata \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d '"hello world"'
```

#### JSON Number

```bash
curl -X POST http://localhost:8000/get_request_metadata \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d '42'
```

#### JSON Boolean

```bash
curl -X POST http://localhost:8000/get_request_metadata \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer token123' \
  -d 'true'
```
