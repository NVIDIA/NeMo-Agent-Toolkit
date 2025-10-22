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

# Simple MCP Authentication Example

This example demonstrates how to use the NVIDIA NeMo Agent toolkit with MCP servers that require authentication. You'll authenticate with protected MCP services and access secured tools through OAuth2 flows.

It is recommended to read the [MCP Authentication](../../../docs/source/workflows/mcp/mcp-auth.md) documentation first.

## Prerequisites

1. **Agent toolkit**: Ensure you have the Agent toolkit installed. If you have not already done so, follow the instructions in the [Install Guide](../../../docs/source/quick-start/installing.md#install-from-source) to create the development environment and install NeMo Agent Toolkit.
2. **MCP Server**: Access to an MCP server that requires authentication (e.g., corporate Jira system)

**Note**: If you installed NeMo Agent toolkit from source, MCP client functionality is already included. If you installed from PyPI, you may need to install the MCP client package separately with `uv pip install "nvidia-nat[mcp]"`.

## Install this Workflow

Install this example:

```bash
uv pip install -e examples/MCP/simple_auth_mcp
```

## Run the Workflow

### Authenticated MCP Client

You can run the workflow using authenticated MCP tools. In this case, the workflow acts as an MCP client and connects to a protected MCP server requiring OAuth2 authentication.

**Prerequisites:**
1. **Set up environment variables**: Configure the required environment variables for your OAuth2 server:
   ```bash
   export CORPORATE_MCP_JIRA_URL="https://your-jira-server.com/mcp"
   ```

   > [!IMPORTANT]
   > Set `CORPORATE_MCP_JIRA_URL` to your protected Jira MCP server URL, not the sample URL shown above. The sample URL is for demonstration purposes only and will not work with your actual Jira instance.

2. **Start the authentication flow**: The first time you run the workflow, it will initiate an OAuth2 authentication flow:
   ```bash
   nat run --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira.yml --input "What is ticket AIQ-1935 about"
   ```

Follow the browser-based authentication flow to authorize access to the MCP server.
3. Example output:
```text
Workflow Result:
['Ticket AIQ-1935 is about converting the experimental function "mcp_client" to function groups. The changes are documented in PR-814 on GitHub. The ticket is currently in the "Done" status.']
```


## Using the Workflow via FastAPI frontend
1. **Start the workflow**:
```bash
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira.yml
```

2. **Start the UI**:

   Start the UI by following the instructions in the [Launching the UI](../../../docs/source/quick-start/launching-ui.md) guide. Connect to the URL http://localhost:3000.

   > [!IMPORTANT]
   > Ensure that `WebSocket` mode is enabled by navigating to the top-right corner and selecting the `WebSocket` option in the arrow pop-out. WebSocket connections are required for OAuth authentication workflows.

3. **Send the input to the workflow via the UI**:
```text
What is ticket AIQ-1935 about
```

## Authentication Flow

1. **Initial Request**: When you first run the workflow, it detects that authentication is required
2. **OAuth2 Redirect**: The system opens your browser to the OAuth2 authorization server
3. **User Authorization**: You log in and authorize the NeMo Agent toolkit to access the MCP server
4. **Token Exchange**: The system exchanges the authorization code for access and refresh tokens
5. **Authenticated Access**: Subsequent requests use the stored tokens to access protected tools
6. **Token Refresh**: Tokens are automatically refreshed when they expire

When using websocket mode you will see two authorization prompts, one for setting up the MCP client and one for `tool calls`. Authorizations for `tool calls` is done per WebSocket session to limit tool access by user.
