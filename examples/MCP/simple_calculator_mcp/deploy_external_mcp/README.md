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

# MCP Server Example

This example demonstrates how to set up and run an MCP (Model Control Protocol) server using a reusable `Dockerfile`.

## Prerequisites

- Docker
- Docker Compose

## Available MCP Services

This example uses the `mcp-server-time` service. For a list of available public MCP services, please refer to the [MCP Server GitHub repository](https://github.com/modelcontextprotocol/servers).

## Setup

1. Change the service name and brief name to the service you want to use. Additionally specify any optional service arguments.

```bash
# This should be the name of the MCP service you want to use.
export SERVICE_NAME=mcp-server-time
# This can be any name you want to give to your service.
export SERVICE_BRIEF_NAME=time
# Any arguments to pass to the service. Example: `mcp-server-time` requires --local-timezone if the container's timezone hasn't been configured.
export SERVICE_ARGS=--"local-timezone \"America/New_York\""

```

2. Set the service directory, server port, and container name.

```bash
export SERVICE_DIR=./.tmp/mcp/${SERVICE_BRIEF_NAME}_service
export CONTAINER_NAME=mcp-proxy-aiq-${SERVICE_BRIEF_NAME}
export SERVER_PORT=8080
```

3. Create a directory for your service and copy the `Dockerfile` to it:

```bash
mkdir -p ${SERVICE_DIR}
cp examples/MCP/simple_calculator_mcp/deploy_external_mcp/Dockerfile ${SERVICE_DIR}/
```

4. Create the run script:

```bash
cat > ${SERVICE_DIR}/run_service.sh <<EOF
#!/bin/bash
uvx ${SERVICE_NAME} ${SERVICE_ARGS}
EOF

chmod +x ${SERVICE_DIR}/run_service.sh
```

5. Create a `docker-compose.yml` file:

```bash
cat > ${SERVICE_DIR}/docker-compose.yml <<EOF
services:
  ${SERVICE_BRIEF_NAME}_server:
    container_name: ${CONTAINER_NAME}
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${SERVER_PORT}:${SERVER_PORT}"
    volumes:
      - ./run_service.sh:/scripts/run_service.sh
    command:
      - "--sse-port=${SERVER_PORT}"
      - "--sse-host=0.0.0.0"
      - "/scripts/run_service.sh"
EOF
```

6. Build and run the MCP server:

```bash
docker compose -f ${SERVICE_DIR}/docker-compose.yml up -d
```

7. Verify the MCP server is running and monitor the service:

```bash
docker ps
docker logs -f ${CONTAINER_NAME}
```

## Usage

The MCP server will be available at `http://localhost:${SERVER_PORT}/sse`. You can use it with any MCP-compatible client.

## Client Configuration

To use the MCP service in your AIQ toolkit application, configure the MCP tool wrapper in your config file:

```yaml
functions:
  ${SERVICE_BRIEF_NAME}_tool:
    _type: mcp_tool_wrapper
    url: "http://0.0.0.0:${SERVER_PORT}/sse"
    mcp_tool_name: get_current_time
    description: "Returns the current date and time from the MCP server"
```

Replace `get_current_time` with the actual tool name exposed by your service.
