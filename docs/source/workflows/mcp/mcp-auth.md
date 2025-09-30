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

# NeMo Agent Toolkit MCP Authentication
MCP provides authorization capabilities at the transport level, enabling MCP clients to make requests to restricted MCP servers on behalf of resource owners. NeMo Agent toolkit provides a set of built-in authentication providers for accessing servers that require authentication. `mcp_oauth2` is the default authentication provider in NeMo Agent toolkit for MCP servers. It conforms to the [MCP OAuth2](https://modelcontextprotocol.io/specification/draft/basic/authorization) specification.

## Authentication Capabilities
NeMo Agent toolkit MCP authentication provides the capabilities required to access protected MCP servers:
- *Dynamic endpoints discovery* using the procedures defined in [RFC9728](https://www.rfc-editor.org/rfc/rfc9728), [RFC 8414](https://www.rfc-editor.org/rfc/rfc8414) and [OpenID Connect](https://openid.net/specs/openid-connect-core-1_0.html)
- *Client registration* using the procedures defined in [RFC7591](https://www.rfc-editor.org/rfc/rfc7591)
- *Authentication* using the procedures defined in the [OAuth2 Specification](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-13)

## Authentication Providers
`mcp_oauth2`is a built-in authentication provider in NeMo Agent toolkit that implements the MCP OAuth2 specification. It is used to authenticate with MCP servers that require authentication.
Sample configuration:
```
authentication:
  auth_provider_mcp:
    _type: mcp_oauth2
    server_url: "http://localhost:9901/mcp"
    redirect_uri: http://localhost:8000/auth/redirect
    default_user_id: ${NAT_USER_ID}
    allow_default_user_id_for_tool_calls: true
```
- `server_url`: The URL of the MCP server that requires authentication.
- `redirect_uri`: The redirect URI for the OAuth2 flow.
- `default_user_id`: The user ID for setting discovering and adding tools to the workflow at startup.
- `allow_default_user_id_for_tool_calls`: Whether to allow the default user ID for tool calls. This is typically enabled for single user workflow, for example a workflow that is launched using `nat run` command. For multi user workflow, this should be disabled.

To view all configuration options for the `mcp_oauth2` authentication provider, run the following command:
```bash
 nat info components -t auth_provider -q mcp_oauth2
```

## MCP Client Configuration
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

## Supported modes
- MCP Authentication is only supported for `streamable-http` transport. It is not supported for local `stdio` transport or for `sse` transport.
- Authentication configuration is only available with `mcp_client` style configuration, not with `mcp_tool_wrapper` style configuration.

## MCP Authentication Example Workflow
The [MCP Authentication Example Workflow](../../../examples/MCP/simple_auth_mcp/README.md) provides an example of how to use the `mcp_oauth2` authentication provider to authenticate with a MCP server.
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
    default_user_id: ${CORPORATE_MCP_JIRA_URL}
    allow_default_user_id_for_tool_calls: ${ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS:-true}
```
### Running the Workflow in Single User Mode (CLI)
In this mode the `default_user_id` is used for authentication during setup and tool calls.
```bash
nat run --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira.yml --input "What is jira ticket AIQ-1935 about"
```

### Running the Workflow in Multi User Mode (FastAPI)
In this mode the `default_user_id` is used for authentication during setup. Subsequent tool calls require each user to authenticate separately via the `Websocket` UI.

1. **Start the workflow**:
```bash
nat serve --config_file examples/MCP/simple_auth_mcp/configs/config-mcp-auth-jira.yml
```
At this point a consent window will be displayed to the user. The user needs to authorize the workflow to access the MCP server. This user's information is cached as the `default user id`. Subsequent tool calls can be disabled for the default user id by setting `allow_default_user_id_for_tool_calls` to `false` in the authentication configuration.

2. **Start the UI**:
Start UI and connect to the URL `http://localhost:3000`. Ensure that `Websocket` mode is enabled by navigating to the top-right corner and selecting the `Websocket` option in the arrow pop-out.

3. **Send the input to the workflow via the UI**:
```text
What is ticket AIQ-1935 about
```
At this point a consent window will again be displayed to the user. The user needs to authorize the workflow to access the MCP server and call the tool. This user's information is cached separately using the session in in the `Websocket` session cookie as the `user id`.

## Displaying Protected MCP Tools via CLI

To use a protected MCP server, you need to provide the `--auth` flag:
```bash
nat mcp client tool list --url http://example.com/mcp --auth
```
This will use the `mcp_oauth2` authentication provider to authenticate the user.

### Troubleshooting
1. If setup fails the `default_user_id` did not complete the authentication flow via the `popup` UI or the user did not authorize the workflow to access the MCP server.
2. If you encounter an error like `User is not authorized to call the tool`, it means that:
- Workflow was not accessed in `Websocket` mode (or)
- User did not complete the authentication flow via the `Websocket` UI (or)
- User is not authorized to call the tool
