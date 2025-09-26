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

# 🧪 Temporary MCP Auth Planning Notes (Experimental)
This is currently a scratch pad for planning MCP auth. It will be rewritten when all the components are ready.

## Phases of MCP auth implementation
1. [Completed] MCP client with new `mcp_oauth2` auth provider.
2. [Pending] MCP protected server with `TokenVerifier`.
3. [Pending] Changes for end-to-end MCP auth testing.

## Steps for testing MCP auth
1. Start the MCP server with auth enabled
```bash
nat mcp serve --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-server-auth.yml
```
This starts a protected MCP server on port 9901. This MCP server has a stub token verifier that will always return success without AS introspection.

2. Start a container with the example auth server from the MCP repo. This will start the auth server on port 9000.
```bash
docker build -t mcp-sample-as -f examples/MCP/simple_calculator_mcp/deploy_example_as/Dockerfile examples/MCP/simple_calculator_mcp/deploy_example_as/
docker run -p 9000:9000 mcp-sample-as
```
This starts the auth server on port 9000.

3. Start NAT UI and enable websocket

4. Run the workflow with MCP auth enabled client

```bash
nat serve --config_file examples/MCP/simple_calculator_mcp/configs/config-mcp-auth-dynamic.yml
```
This starts the workflow with a MCP client that uses the `mcp_oauth2` auth provider. This provider:
- Discovers the auth server endpoints.
- Registers a client with the auth server.
- Performs the OAuth2 authorization code flow using the `OAuth2AuthCodeFlowProvider`.
