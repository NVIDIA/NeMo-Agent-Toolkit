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

The NeMo Agent toolkit supports four different authentication approaches for MCP clients, providing flexibility for various deployment scenarios and security requirements. Authentication is configured using NAT's authentication provider system with `AuthenticationRef` references.

### Option 1: No Authentication (Default)
For public MCP servers that don't require authentication. Simply omit the `auth_provider` field.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://public-mcp-server.com/mcp
    # No auth_provider = no authentication
```

### Option 2: Manual Registration + Explicit Auth Server
Use NAT's existing `oauth2_auth_code_flow` provider for standard OAuth2 authentication.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://protected-mcp-server.com/mcp
      auth_provider: standard_oauth2

authentication:
  standard_oauth2:
    _type: oauth2_auth_code_flow
    authorization_url: https://auth.example.com/oauth/authorize
    token_url: https://auth.example.com/oauth/token
    client_id: ${MCP_CLIENT_ID}
    client_secret: ${MCP_CLIENT_SECRET}
    redirect_uri: http://localhost:8000/auth/redirect
    scopes: ["mcp:read", "mcp:write"]
    use_pkce: true
```

### Option 3: Dynamic Registration + MCP Server Discovery
Use the new `mcp_oauth2` provider for fully automated discovery and dynamic client registration.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://protected-mcp-server.com/mcp
      auth_provider: mcp_oauth2_dynamic

authentication:
  mcp_oauth2_dynamic:
    _type: mcp_oauth2
    enable_dynamic_registration: true
    redirect_uri: http://localhost:8000/auth/redirect
    client_name: "NAT MCP Client"
    # All auth details discovered from MCP server
    # Client credentials obtained via dynamic registration
```

### Option 4: Manual Registration + MCP Server Discovery (Hybrid)
Use the new `mcp_oauth2` provider with pre-registered client credentials and MCP server discovery.

```yaml
functions:
  my_mcp_client:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://protected-mcp-server.com/mcp
      auth_provider: mcp_oauth2_hybrid

authentication:
  mcp_oauth2_hybrid:
    _type: mcp_oauth2
    client_id: ${MCP_CLIENT_ID}
    client_secret: ${MCP_CLIENT_SECRET}
    redirect_uri: http://localhost:8000/auth/redirect
    client_name: "NAT MCP Client"
    # Auth server URLs and scopes discovered from MCP server
    # authorization_url: <-- discovered via RFC 9728 + RFC 8414
    # token_url: <-- discovered via RFC 9728 + RFC 8414
    # scopes: <-- discovered from MCP server metadata
```

## Configuration Details

### Required Fields by Option

#### **Option 1 (No Authentication)**
- No `auth_provider` field in MCP client configuration

#### **Option 2 (Manual + Explicit)**
- `auth_provider`: Reference to `oauth2_auth_code_flow` provider
- Provider must include:
  - `authorization_url`: OAuth2 authorization endpoint URL
  - `token_url`: OAuth2 token endpoint URL
  - `client_id`: OAuth2 client identifier
  - `client_secret`: OAuth2 client secret

#### **Option 3 (Dynamic + Discovery)**
- `auth_provider`: Reference to `mcp_oauth2` provider
- Provider must include:
  - `enable_dynamic_registration`: Set to `true` to enable automatic client registration

#### **Option 4 (Hybrid)**
- `auth_provider`: Reference to `mcp_oauth2` provider
- Provider must include:
  - `client_id`: OAuth2 client identifier (pre-registered)
  - `client_secret`: OAuth2 client secret (pre-registered)

### Optional Fields (All Options)
- `redirect_uri`: OAuth2 callback URL (defaults to localhost with random port)
- `scopes`: Required OAuth2 scopes (discovered from MCP server if not provided)
- `use_pkce`: Enable PKCE for authorization code flow (default: true)
- `client_name`: OAuth2 client name for dynamic registration (default: "NAT MCP Client")

### Discovery Process
When auth server details are not explicitly provided, the client will:

1. **MCP Server Discovery**: Make initial request to MCP server
   - **MCP-SDK**: Handles 401 response detection and WWW-Authenticate header parsing
   - **NAT**: Provides server URL and transport configuration

2. **Resource Metadata**: Fetch `/.well-known/oauth-protected-resource` from MCP server
   - **MCP-SDK**: Automatically fetches and parses resource metadata (RFC 9728)
   - **NAT**: No additional work required

3. **Auth Server Discovery**: Fetch `/.well-known/oauth-authorization-server` from auth server
   - **MCP-SDK**: Automatically fetches and parses auth server metadata (RFC 8414)
   - **NAT**: No additional work required

4. **Dynamic Registration**: Optionally register client with auth server (RFC 7591)
   - **MCP-SDK**: Handles complete dynamic client registration flow
   - **NAT**: Provides client metadata (name, redirect URIs, etc.)

5. **Authorization Flow**: Perform OAuth2 authorization code flow with PKCE
   - **MCP-SDK**: Handles complete OAuth2 flow including PKCE, token exchange, refresh
   - **NAT**: Provides redirect URI and callback handling

## Token Management

Token management handles the lifecycle of OAuth2 access tokens, including storage, refresh, and expiration handling.

### Token Storage
- **NAT Integration**: Tokens stored using NAT's existing credential management system
- **Secure Storage**: Leverages NAT's encrypted credential storage for client secrets and access tokens
- **Per-Client Isolation**: Each MCP client configuration maintains separate token storage

### Token Refresh
- **Automatic Refresh**: MCP-SDK handles automatic token refresh using refresh tokens
- **Background Processing**: Token refresh occurs transparently during MCP operations
- **Error Handling**: Failed refresh attempts trigger re-authentication flows

### Token Expiration
- **Proactive Refresh**: Tokens refreshed before expiration to avoid service interruptions
- **Graceful Degradation**: Expired tokens trigger automatic re-authentication
- **User Notification**: Clear error messages when manual re-authentication is required

### Integration with NAT Auth System
- **Credential Provider**: MCP client tokens integrate with NAT's existing `CredentialProvider` system
- **Environment Variables**: Support for `${TOKEN_VAR}` style token references
- **Secrets Management**: Compatible with NAT's secrets management for production deployments

## MCP Server Configuration

For MCP clients to discover authorization servers (Options 3 and 4), the MCP server must be configured with auth requirements.

### NAT MCP Server Configuration

```yaml
general:
  front_end:
    _type: mcp
    host: localhost
    port: 9091
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
