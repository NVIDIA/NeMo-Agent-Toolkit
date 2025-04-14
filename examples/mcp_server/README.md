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

This example demonstrates how to set up and run an MCP (Model Control Protocol) server using a reusable Dockerfile.

## Prerequisites

- Docker and Docker Compose


## Available MCP Services

The sample below uses the `mcp-server-time` service. For a list of available public MCP services, please refer to the [MCP Server GitHub repository](https://github.com/modelcontextprotocol/servers).


## Setup

1. Set service directory and name(using `mcp-server-time` as an example):
``` bash
export SERVICE_DIR=./.tmp/mcp/time_service
```

2. Create a directory for your service:
``` bash
mkdir -p ${SERVICE_DIR}
```

2. Copy the Dockerfile to your service directory:
```bash
cp examples/mcp_server/Dockerfile ${SERVICE_DIR}/
```

3. Create the run script:
```bash
cat > ${SERVICE_DIR}/run_service.sh <<EOF
#!/bin/bash
uvx run mcp-server-time
EOF

chmod +x ${SERVICE_DIR}/run_service.sh
```

4. Create a docker-compose.yml file:
```bash
cat > ${SERVICE_DIR}/docker-compose.yml <<EOF
services:
  mcp_time_server:
    container_name: mcp-proxy-aiq-time
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    volumes:
      - ./run_service.sh:/scripts/run_service.sh
    command:
      - "--sse-port=8080"
      - "--sse-host=0.0.0.0"
      - "/scripts/run_service.sh"
EOF
```

5. Build and run the MCP server:
```bash
docker compose -f ${SERVICE_DIR}/docker-compose.yml up -d
```

## Usage

The MCP server will be available at `http://localhost:8080/sse`. You can use it with any MCP-compatible client.

## Client Configuration

To use the MCP service in your AgentIQ application, configure the MCP tool wrapper in your config file:

```yaml
functions:
  mcp_time_tool:
    _type: mcp_tool_wrapper
    url: "http://0.0.0.0:8080/sse"
    mcp_tool_name: time
    description: "Returns the current date and time from the MCP server"
```
