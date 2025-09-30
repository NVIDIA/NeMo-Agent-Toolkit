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

# MCP Authentication for the NVIDIA NeMo Agent toolkit
MCP provides authorization capabilities at the transport level, enabling MCP clients to make requests to restricted MCP servers on behalf of resource owners. The NVIDIA NeMo Agent toolkit provides a set of built-in authentication providers for accessing servers that require authentication. The `mcp_oauth2` provider is the default authentication provider in the NeMo Agent toolkit for MCP servers. It conforms to the [MCP OAuth2](https://modelcontextprotocol.io/specification/draft/basic/authorization) specification.

## Supported Capabilities
NeMo Agent toolkit MCP authentication provides the capabilities required to access protected MCP servers:
- Dynamic endpoint discovery using the procedures defined in [RFC 9728](https://www.rfc-editor.org/rfc/rfc9728), [RFC 8414](https://www.rfc-editor.org/rfc/rfc8414), and [OpenID Connect](https://openid.net/specs/openid-connect-core-1_0.html)
- Client registration using the procedures defined in [RFC 7591](https://www.rfc-editor.org/rfc/rfc7591)
- Authentication using the procedures defined in the [OAuth2 specification](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-13)

## Configuring an Auth Provider
`mcp_oauth2` is a built-in authentication provider in the NeMo Agent toolkit that implements the MCP OAuth2 specification. It is used to authenticate with MCP servers that require authentication.
Sample configuration:
```yaml
authentication:
  auth_provider_mcp:
    _type: mcp_oauth2
    server_url: "http://localhost:9901/mcp"
    redirect_uri: http://localhost:8000/auth/redirect
    default_user_id: ${NAT_USER_ID}
    allow_default_user_id_for_tool_calls: ${ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS:-true}
```
Configuration options:
- `server_url`: The URL of the MCP server that requires authentication.
- `redirect_uri`: The redirect URI for the OAuth2 flow.
- `default_user_id`: The user ID for discovering and adding tools to the workflow at startup. The `default_user_id` can be any string and is used as the key to cache the user's information. It defaults to the `server_url` if not provided.
- `allow_default_user_id_for_tool_calls`: Whether to allow the default user ID for tool calls. This is typically enabled for single-user workflows, for example, a workflow that is launched using the `nat run` CLI command. For multi-user workflows, this should be disabled to avoid accidental tool calls by unauthorized users.

To view all configuration options for the `mcp_oauth2` authentication provider, run the following command:
```bash
 nat info components -t auth_provider -q mcp_oauth2
```

### Environment Variables
Some configuration values are commonly provided through environment variables:
- `NAT_USER_ID`: Used as `default_user_id` to cache the authenticating user during setup and optionally for tool calls. Defaults to the `server_url` if not provided.
- `ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS`: Controls whether the default user can invoke tools. Defaults to `true` if not provided.

Set them for your current shell:
```bash
export NAT_USER_ID="dev-user"
export ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS=true
```
## Referencing Auth Providers in Clients
The authentication provider is referenced by name via the `auth_provider` parameter in the MCP client configuration.
```yaml
function_groups:
  mcp_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: "http://localhost:9901/mcp"
      auth_provider: auth_provider_mcp
```

## Limitations & Supported Transports
- MCP Authentication is only supported for `streamable-http` transport. It is not supported for local `stdio` transport or for `sse` transport.
- Authentication configuration is only available with `mcp_client` style configuration, not with `mcp_tool_wrapper` style configuration.

## Example Workflow
The MCP Authentication Example Workflow, `examples/MCP/simple_auth_mcp/README.md`, provides an example of how to use the `mcp_oauth2` authentication provider to authenticate with a MCP server.
### Example Configuration
```yaml
function_groups:
  mcp_jira:
    _type: mcp_client
    server:
      transport: streamable-http
      url: ${CORPORATE_MCP_JIRA_URL}
    auth_provider: mcp_oauth2_jira

authentication:
  mcp_oauth2_jira:
    _type: mcp_oauth2
    server_url: ${CORPORATE_MCP_JIRA_URL}
    redirect_uri: http://localhost:8000/auth/redirect
    default_user_id: ${NAT_USER_ID}
    allow_default_user_id_for_tool_calls: ${ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS:-true}
```
### Running the Workflow in Single-User Mode (CLI)
In this mode, the `default_user_id` is used for authentication during setup and for subsequent tool calls.

```{mermaid}
flowchart LR
  U[User<br/>default-user-id] --> H[MCP Host<br/>NAT Workflow]
  H --> C[MCP Client<br/>default-user-id]
  C --> S[MCP Server<br/>Protected Jira Service]
```

Set the environment variables to access the protected MCP server:
```bash
export CORPORATE_MCP_JIRA_URL="https://your-jira-server.com/mcp"
export NAT_USER_ID="dev-user"
export ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS=true
```
Then run the workflow:
```bash
nat run --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira.yml --input "What is Jira ticket AIQ-1935 about"
```

### Running the Workflow in Multi-User Mode (FastAPI)
In this mode the workflow is served via a FastAPI frontend. Multiple users can access the workflow concurrently using a UI with `WebSocket` mode enabled.

```{mermaid}
flowchart LR
  U0[User<br/>default-user-id] --> H2[MCP Host<br/>NAT Workflow]
  U1[User<br/>UI-User-1] --> H2
  U2[User<br/>UI-User-2] --> H2

  H2 --> C0[MCP Client<br/>default-user-id]
  H2 --> C1[MCP Client<br/>UI-User-1]
  H2 --> C2[MCP Client<br/>UI-User-2]

  C0 --> S2[MCP Server]
  C1 --> S2
  C2 --> S2
```

1. Set the environment variables to access the protected MCP server:
```bash
export CORPORATE_MCP_JIRA_URL="https://your-jira-server.com/mcp"
export NAT_USER_ID="dev-user"
export ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS=false
```
2. **Start the workflow**:
```bash
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira.yml
```
At this point, a consent window is displayed to the user. The user must authorize the workflow to access the MCP server. This user's information is cached as the default user ID. The `default_user_id` credentials are only used for the initial setup and for populating the tools in the workflow or agent prompt at startup.

Subsequent tool calls can be disabled for the default user ID by setting `allow_default_user_id_for_tool_calls` to `false` in the authentication configuration. This is recommended for multi-user workflows to avoid accidental tool calls by unauthorized users.

3. **Start the UI**:
- Start the UI by following the instructions in the [User Interface](../../quick-start/launching-ui.md) documentation.
- Connect to  the UI at `http://localhost:3000`
- Ensure that `WebSocket` mode is enabled by navigating to the top-right corner and selecting the `WebSocket` option in the arrow pop-out.

4. **Send the input to the workflow via the UI**:
```text
What is ticket AIQ-1935 about
```
At this point, a consent window is displayed again. The `UI` user must authorize the workflow to access the MCP server and call the tool. This user's information is cached separately using the `WebSocket` session cookie as the user ID.

## Displaying Protected MCP Tools via CLI
MCP client CLI can be used to display and call MCP tools on a remote MCP server. To use a protected MCP server, you need to provide the `--auth` flag:
```bash
nat mcp client tool list --url http://example.com/mcp --auth
```
This will use the `mcp_oauth2` authentication provider to authenticate the user. For more information, see the [MCP Client](./mcp-client.md) documentation.

## Security Considerations
- The `default_user_id` is used to cache the authenticating user during setup and optionally for tool calls. It is recommended to set `allow_default_user_id_for_tool_calls` to `false` in the authentication configuration for multi-user workflows to avoid accidental tool calls by unauthorized users.
- Use HTTPS redirect URIs in production environments.
- Scope OAuth2 tokens to the minimum required permissions.

## Troubleshooting
1.  **Setup fails** - This can happen if:
- The user identified by `default_user_id` did not complete the authentication flow through the pop-up UI, or
- The user did not authorize the workflow to access the MCP server

2. **Tool calls fail** - This can happen if:
- The workflow was not accessed in `WebSocket` mode, or
- The user did not complete the authentication flow through the `WebSocket` UI, or
- The user is not authorized to call the tool
