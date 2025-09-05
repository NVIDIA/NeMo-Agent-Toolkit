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

# Temporary MCP Auth Planning Notes
This is currently a scratch pad for planning MCP auth. It will be removed when MCP auth is implemented.

## MCP Client Configuration Options

The NeMo Agent toolkit supports four different authentication approaches for MCP clients, providing flexibility for various deployment scenarios and security requirements.

### Option 1: No Authentication (Default)
For public MCP servers that don't require authentication.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://public-mcp-server.com/mcp
      # No auth configuration needed
```

### Option 2: Manual Registration + Explicit Auth Server
Complete manual configuration similar to NAT's existing OAuth2 pattern.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://protected-mcp-server.com/mcp
      auth:
        enabled: true
        # Explicit auth server configuration (similar to NAT's oauth2_auth_code_flow)
        authorization_url: https://auth.example.com/oauth/authorize
        token_url: https://auth.example.com/oauth/token
        client_id: ${MCP_CLIENT_ID}
        client_secret: ${MCP_CLIENT_SECRET}
        redirect_uri: http://localhost:8080/oauth/callback
        scopes: ["mcp:read", "mcp:write"]
        use_pkce: true
```

### Option 3: Dynamic Registration + MCP Server Discovery
Fully automated approach using MCP specification discovery and dynamic client registration.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://protected-mcp-server.com/mcp
      auth:
        enabled: true
        # All auth details discovered from MCP server
        # Client credentials obtained via dynamic registration
        enable_dynamic_registration: true
        redirect_uri: http://localhost:8080/oauth/callback
```

### Option 4: Manual Registration + MCP Server Discovery (Hybrid)
Pre-registered client credentials with MCP-compliant discovery of auth server details.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://protected-mcp-server.com/mcp
      auth:
        enabled: true
        # Pre-registered client credentials
        client_id: ${MCP_CLIENT_ID}
        client_secret: ${MCP_CLIENT_SECRET}
        # Auth server URLs and scopes discovered from MCP server
        # authorization_url: <-- discovered via RFC 9728 + RFC 8414
        # token_url: <-- discovered via RFC 9728 + RFC 8414
        # scopes: <-- discovered from MCP server metadata
        redirect_uri: http://localhost:8080/oauth/callback
```

## Configuration Details

### Required Fields
- `enabled`: Set to `true` to enable OAuth2 authentication
- `redirect_uri`: OAuth2 callback URL for authorization code flow

### Optional Fields
- `authorization_server_url`: Explicit auth server URL (discovered if not provided)
- `client_id`: OAuth2 client identifier (obtained via dynamic registration if not provided)
- `client_secret`: OAuth2 client secret (obtained via dynamic registration if not provided)
- `scopes`: Required OAuth2 scopes (discovered from MCP server if not provided)
- `enable_dynamic_registration`: Enable RFC 7591 dynamic client registration (default: true)
- `enable_pkce`: Enable PKCE for authorization code flow (default: true)

### Discovery Process
When auth server details are not explicitly provided, the client will:

1. **MCP Server Discovery**: Make initial request to MCP server
2. **Resource Metadata**: Fetch `/.well-known/oauth-protected-resource` from MCP server
3. **Auth Server Discovery**: Fetch `/.well-known/oauth-authorization-server` from auth server
4. **Dynamic Registration**: Optionally register client with auth server (RFC 7591)
5. **Authorization Flow**: Perform OAuth2 authorization code flow with PKCE

## MCP Server Configuration

For MCP clients to discover authorization servers (Options 3 and 4), the MCP server must be configured with auth requirements.

### NAT MCP Server Configuration

```yaml
general:
  front_end:
    _type: mcp
    host: 0.0.0.0
    port: 8080
    # Auth configuration for MCP server
    require_auth: true
    auth_server_url: https://auth.example.com
    required_scopes: ["mcp:read", "mcp:write"]
```

### What This Enables

When configured, the NAT MCP server will:

1. **Return 401 responses** to the MCP client for unauthenticated requests with WWW-Authenticate header:
   ```
   WWW-Authenticate: Bearer realm="mcp", authorization_uri="https://auth.example.com"
   ```

2. **Serve resource metadata** to the MCP client at `/.well-known/oauth-protected-resource`:
   ```json
   {
     "authorization_servers": ["https://auth.example.com"],
     "scopes_supported": ["mcp:read", "mcp:write"]
   }
   ```

3. **Validate Bearer tokens** from MCP clients using the configured authorization server. Yet to be implemented.

### Authorization Server Requirements

The authorization server (`https://auth.example.com`) must support both client-facing and server-facing APIs:

#### **Client-Facing APIs** (used by MCP Client)

1. **Support RFC 8414** - Serve metadata at `/.well-known/oauth-authorization-server`:
   ```json
   {
     "authorization_endpoint": "https://auth.example.com/oauth/authorize",
     "token_endpoint": "https://auth.example.com/oauth/token",
     "registration_endpoint": "https://auth.example.com/register",
     "scopes_supported": ["mcp:read", "mcp:write"],
     "response_types_supported": ["code"],
     "grant_types_supported": ["authorization_code", "refresh_token"]
   }
   ```

2. **Support RFC 7591 dynamic client registration** (required for Option 3):
   - **Registration endpoint**: Accept POST requests to `/register` with client metadata
   - **Automatic credential generation**: Return `client_id` and `client_secret` for new clients
   - **Client metadata support**: Handle `client_name`, `redirect_uris`, `grant_types`, `scopes`
   - **Example registration request**:
     ```json
     POST /register
     {
       "client_name": "NAT MCP Client",
       "redirect_uris": ["http://localhost:8080/oauth/callback"],
       "grant_types": ["authorization_code", "refresh_token"],
       "response_types": ["code"],
       "scope": "mcp:read mcp:write"
     }
     ```
   - **Example registration response**:
     ```json
     {
       "client_id": "dynamically-generated-client-123",
       "client_secret": "auto-generated-secret-456",
       "client_id_issued_at": 1640995200,
       "client_secret_expires_at": 1672531200
     }
     ```

#### **Server-Facing APIs** (used by MCP Server)

3. **Support token introspection** - Validate Bearer tokens from MCP clients:
   - **Introspection endpoint**: Accept POST requests to validate access tokens
   - **Token validation**: Return token status, scopes, client information
   - **Example introspection request**:
     ```json
     POST /introspect
     Authorization: Basic <mcp_server_credentials>
     Content-Type: application/x-www-form-urlencoded

     token=ACCESS_TOKEN_456&token_type_hint=access_token
     ```
   - **Example introspection response**:
     ```json
     {
       "active": true,
       "client_id": "dynamically-generated-client-123",
       "scope": "mcp:read mcp:write",
       "exp": 1672531200,
       "aud": ["https://mcp-server.com"]
     }
     ```
