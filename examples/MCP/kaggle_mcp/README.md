<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

This example demonstrates how to use the Kaggle MCP server with NVIDIA NeMo Agent Toolkit to interact with Kaggle's datasets, notebooks, models, and competitions.

## Prerequisites

- NeMo Agent Toolkit installed with MCP support (`nvidia-nat-mcp` package)
- A Kaggle account and API token

### Getting Your Kaggle Bearer Token

The Kaggle MCP server uses bearer token authentication. Obtain your Kaggle bearer token from [Kaggle Account Settings](https://www.kaggle.com/settings/account).

## Configuration

The `config.yml` file uses the built-in `api_key` authentication provider with Bearer token scheme:

```yaml
authentication:
  kaggle:
    _type: api_key
    raw_key: ${KAGGLE_BEARER_TOKEN}
    auth_scheme: Bearer
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
  --input "list the IMDB datasets"
```

Example queries:
- "What is the titanic dataset about?"
- "What competitions are currently active?"

### Per-User Mode (Multi-User Server)

For multi-user deployments where each user needs their own isolated workflow and MCP client instance, use the per-user configuration:

```bash
export KAGGLE_BEARER_TOKEN="your_kaggle_api_key_here"
nat serve --config_file examples/MCP/kaggle_mcp/configs/config-per-user.yml
```

Test requests with different users:

User Alice:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=user-alice" \
  -d '{"messages": [{"role": "user", "content": "Search for titanic datasets"}]}'
```

User Bob (has a separate MCP client instance):
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=user-bob" \
  -d '{"messages": [{"role": "user", "content": "What competitions are active?"}]}'
```

Each user identified by their `nat-session` cookie gets their own workflow instance and MCP client.

## Configuration Details

### MCP Client Setup

The configuration connects to Kaggle's MCP server using:
- **Transport**: `streamable-http` (recommended for HTTP-based MCP servers)
- **URL**: `https://www.kaggle.com/mcp`
- **Authentication**: Bearer token via the built-in `api_key` authentication provider

## CLI Commands

You can use the following CLI commands to interact with the Kaggle MCP server. This is useful for prototyping and debugging.

### Discover Tools (No Authentication Required)

To list available tools from the Kaggle MCP server:

```bash
nat mcp client tool list --url https://www.kaggle.com/mcp
```

### Get Tool Schema (No Authentication Required)

To validate the tool schema:

```bash
nat mcp client tool list --url https://www.kaggle.com/mcp --tool search_datasets
```

### Authenticated Tool Calls

The Kaggle MCP server requires bearer token authentication for some tool calls.

#### Using Environment Variable (Recommended)

```bash
# Set your Kaggle bearer token
export KAGGLE_BEARER_TOKEN="your_kaggle_api_key_here"

# Search for Titanic datasets
nat mcp client tool call search_datasets \
  --url https://www.kaggle.com/mcp \
  --bearer-token-env KAGGLE_BEARER_TOKEN \
  --json-args '{"request": {"search": "titanic"}}'
```

#### Using Direct Token

```bash
# Search for Titanic datasets with direct token (less secure)
nat mcp client tool call search_datasets \
  --url https://www.kaggle.com/mcp \
  --bearer-token "your_kaggle_api_key_here" \
  --json-args '{"request": {"search": "titanic"}}'
```

**Note**: The `--bearer-token-env` approach is more secure because it doesn't expose the token in command history or process lists.

## Troubleshooting

### Agent Uses Wrong Parameter Names

**Problem**: The agent generates tool calls with incorrect parameter names, such as using `query` instead of `search` for `search_datasets`.

**Cause**: The default tool descriptions from Kaggle MCP are generic and don't specify parameter names, causing the LLM to infer incorrect names.

**Solution**: Check the tool schema and add tool overrides in your `config.yml` to provide explicit parameter guidance:

```bash
nat mcp client tool list --url https://www.kaggle.com/mcp --tool search_datasets
```

After getting the tool schema, add the following tool overrides to your `config.yml`:

```yaml
function_groups:
  kaggle_mcp_tools:
    tool_overrides:
      search_datasets:
        description: >
          Search for datasets on Kaggle. Use the 'search' parameter (not 'query')
          to search by keywords. Example: {"request": {"search": "titanic"}}
```

### Permission Denied Errors

**Problem**: Tool calls fail with "Permission 'datasets.get' was denied" or similar errors.

**Cause**: Your Kaggle API token lacks the required permissions for certain operations.

**Solution**:
- Ensure you're using a valid Kaggle API key from https://www.kaggle.com/settings/account
- Some operations require dataset ownership or special permissions
- Use `search_datasets` for browsing (requires minimal permissions)
- Use `list_dataset_files` only for datasets you own or have access to

### CLI Tool Calls Work but Workflow Fails

**Problem**: `nat mcp client tool call` succeeds but `nat run` with a workflow fails with the same tool.

**Possible causes**:
1. **Parameter validation**: CLI bypasses some validation that workflows enforce
2. **Agent parameter inference**: Agent might use wrong parameter names (see "Agent Uses Wrong Parameter Names" above)

**Solution**: Use `--direct` mode to test the raw MCP server behavior, then add tool overrides to guide the agent.

## References

- [Kaggle MCP Documentation](https://www.kaggle.com/docs/mcp)
- [NeMo Agent Toolkit MCP Documentation](../../../docs/source/build-workflows/mcp-client.md)
