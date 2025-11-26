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

# Kaggle MCP Example

This example demonstrates how to use the Kaggle MCP server with NeMo Agent Toolkit (NAT) to interact with Kaggle's datasets, notebooks, models, and competitions.

## Prerequisites

- NAT installed with MCP support (`nvidia-nat-mcp` package)
- A Kaggle account and API token

## Getting Your Kaggle Bearer Token

The Kaggle MCP server uses bearer token authentication. To obtain your token:

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/settings)
2. Scroll down to the **API** section
3. Click **Create New Token**
4. This will download a `kaggle.json` file containing your credentials:
   ```json
   {
     "username": "your_username",
     "key": "your_api_key"
   }
   ```
5. Your bearer token is the `key` value from this file

## Configuration

The `config.yml` file uses NAT's built-in `api_key` authentication provider with Bearer token scheme:

```yaml
authentication:
  kaggle:
    _type: api_key
    raw_key: ${KAGGLE_BEARER_TOKEN}
    auth_scheme: BEARER
```

### Environment Variables

Set the following environment variable:

```bash
export KAGGLE_BEARER_TOKEN="your_kaggle_api_key_here"
```

## Usage

Run the workflow with a query:

```bash
nat run --config_file examples/MCP/kaggle_mcp/configs/config.yml \
  --input "Search for datasets about machine learning"
```

Example queries:
- "Find the most popular datasets about natural language processing"
- "Search for notebooks related to computer vision"
- "What competitions are currently active?"
- "List the files in the titanic dataset"

## Configuration Details

### Authentication Provider

This example uses NAT's `api_key` authentication provider, which supports:
- **BEARER** scheme (default) - Adds `Authorization: Bearer <token>` header
- **X_API_KEY** scheme - Adds `X-Api-Key: <token>` header
- **CUSTOM** scheme - Custom header name and prefix

For Kaggle, we use the BEARER scheme as shown in the Kaggle MCP documentation.

### MCP Client Setup

The configuration connects to Kaggle's MCP server using:
- **Transport**: `streamable-http` (recommended for HTTP-based MCP servers)
- **URL**: `https://www.kaggle.com/mcp`
- **Authentication**: Bearer token via `api_key` provider

## CLI Commands
You can use the following CLI commands to interact with the Kaggle MCP server. This is useful for prototyping and debugging.

### Tool Discovery

To list available tools from the Kaggle MCP server:

```bash
nat mcp client tool list --url https://www.kaggle.com/mcp
```
### Tool schema validation

To validate the tool schema:

```bash
nat mcp client tool list --url https://www.kaggle.com/mcp --tool list_dataset_files
```

### Tool call

To call a tool:

```bash
nat mcp client tool call list_dataset_files --url https://www.kaggle.com/mcp --json-args '{"dataset_name": "titanic"}'
```

## References

- [Kaggle MCP Documentation](https://www.kaggle.com/docs/mcp)
- [NAT MCP Documentation](https://docs.nvidia.com/nemo-agent-toolkit/mcp/)
- [NAT Authentication Guide](https://docs.nvidia.com/nemo-agent-toolkit/authentication/)
