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

# MCP Service Account Authentication

Service account authentication enables headless, automated access to MCP servers using OAuth2 client credentials flow. This authentication method is designed for scenarios where interactive user authentication is not possible or desirable, such as CI/CD pipelines, backend services, and automated workflows.

## When to Use Service Account Authentication

Service account authentication is ideal for:

- **CI/CD Pipelines**: Automated testing and deployment workflows that need to access MCP servers
- **Backend Services**: Server-to-server communication without user interaction
- **Batch Processing**: Scheduled jobs that process data from MCP servers
- **Automated Workflows**: Any scenario where a browser-based OAuth2 flow is not feasible
- **Container Deployments**: Containerized applications that need consistent, non-interactive authentication

Use interactive OAuth2 authentication (`mcp_oauth2`) instead when:
- Users need to authorize access to their personal data
- User-specific permissions are required
- The workflow is user-facing and can present a browser for authentication

## Supported Capabilities

The `mcp_service_account` authentication provider implements:

- **OAuth2 Client Credentials Flow**: Standard [RFC 6749 Section 4.4](https://www.rfc-editor.org/rfc/rfc6749#section-4.4) client credentials grant
- **Token Caching**: Automatic token caching with configurable refresh buffer to minimize token endpoint requests
- **Custom Token Formats**: Support for non-standard Bearer token prefixes (for example, `Bearer service_account_ssa:token`)
- **Multi-Header Authentication**: Ability to inject multiple authentication headers for services requiring additional tokens

## Configuring Service Account Auth Provider

The `mcp_service_account` provider is a built-in authentication provider in the NVIDIA NeMo Agent toolkit. Configure it in your workflow YAML file:

```yaml
authentication:
  my_service_account:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: https://auth.example.com/service_account/token
    scopes: "api.read api.write"
```

To view all configuration options for the `mcp_service_account` authentication provider, run the following command:

```bash
nat info components -t auth_provider -q mcp_service_account
```

### Required Configuration Fields

The following fields must be provided in your configuration:

| Field | Description | Example |
|-------|-------------|---------|
| `client_id` | OAuth2 client identifier for your service account | `my-service-client` |
| `client_secret` | OAuth2 client secret (keep secure, never commit to version control) | `${SERVICE_ACCOUNT_CLIENT_SECRET}` |
| `token_url` | OAuth2 token endpoint URL | `https://auth.example.com/oauth/token` |
| `scopes` | Space-separated list of OAuth2 scopes required for access | `api.read api.write` |

### Optional Configuration Fields

Customize the authentication behavior with these optional fields:

| Field | Default | Description |
|-------|---------|-------------|
| `token_prefix` | `service_account` | Prefix for the Authorization header token. Use empty string (`""`) for standard Bearer tokens. |
| `service_token` | None | Additional service-specific token for dual-header authentication patterns |
| `service_token_header` | `Service-Account-Token` | Header name for the service-specific token |
| `token_cache_buffer_seconds` | `300` | Seconds before token expiry to refresh the token (default: 5 minutes) |

## Environment Variables

Service account credentials are typically provided through environment variables to avoid committing secrets to version control. Reference them in your configuration using the `${VARIABLE_NAME}` syntax:

```yaml
authentication:
  my_service_account:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes: ${SERVICE_ACCOUNT_SCOPES}
```

Set the environment variables in your shell:

```bash
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
export SERVICE_ACCOUNT_CLIENT_SECRET="your-client-secret"
export SERVICE_ACCOUNT_TOKEN_URL="https://auth.example.com/oauth/token"
export SERVICE_ACCOUNT_SCOPES="api.read api.write"
```

:::{warning}
**Security Best Practice**: Never commit credentials to version control. Always use environment variables or a secure secret management system for storing service account credentials.
:::

## Referencing Auth Providers in Clients

Reference the service account authentication provider in your MCP client configuration using the `auth_provider` parameter:

```yaml
function_groups:
  mcp_tools:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://api.example.com/mcp
      auth_provider: my_service_account

authentication:
  my_service_account:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes: ${SERVICE_ACCOUNT_SCOPES}
```


## Authentication Patterns

The service account provider supports multiple authentication patterns to accommodate different service requirements.

### Standard Bearer Token (Single Header)

The simplest configuration uses a standard OAuth2 Bearer token:

```yaml
authentication:
  standard_auth:
    _type: mcp_service_account
    client_id: ${CLIENT_ID}
    client_secret: ${CLIENT_SECRET}
    token_url: https://auth.example.com/oauth/token
    scopes: "api.read"
    token_prefix: ""  # Empty string for standard Bearer token
```

This produces the following header:
```
Authorization: Bearer <access_token>
```

### Custom Token Prefix (Single Header)

Some services require a custom token prefix:

```yaml
authentication:
  custom_auth:
    _type: mcp_service_account
    client_id: ${CLIENT_ID}
    client_secret: ${CLIENT_SECRET}
    token_url: https://auth.example.com/oauth/token
    scopes: "service.scope"
    token_prefix: service_account_ssa
```

This produces the following header:
```text
Authorization: Bearer service_account_ssa:<access_token>
```

### Dual-Header Pattern

Some services require two separate authentication headers:

```yaml
authentication:
  dual_header_auth:
    _type: mcp_service_account
    client_id: ${CLIENT_ID}
    client_secret: ${CLIENT_SECRET}
    token_url: ${TOKEN_URL}
    scopes: "service.scope"
    token_prefix: service_account_ssa
    service_token: ${SERVICE_TOKEN}
    service_token_header: X-Service-Account-Token
```

This produces the following headers:
```text
Authorization: Bearer service_account_ssa:<access_token>
X-Service-Account-Token: <service_token>
```

## Security Considerations

### Credential Management

- **Never commit credentials**: Store credentials in environment variables or secure secret management systems
- **Rotate credentials regularly**: Implement a credential rotation policy
- **Use minimal scopes**: Request only the OAuth2 scopes required for your use case
- **Monitor token usage**: Track token requests and usage patterns for anomalies

### Token Security

- **In-memory only**: Tokens are cached in memory and not persisted to disk
- **Automatic refresh**: Tokens are refreshed before expiration to minimize exposure window
- **HTTPS required**: Always use HTTPS for token endpoint communications
- **Protected logs**: Failed requests do not expose credentials in log messages


## Token Caching

The service account provider implements intelligent token caching to minimize requests to the OAuth2 token endpoint:

1. **First Request**: Client credentials are sent to the token endpoint, and the access token is cached
2. **Subsequent Requests**: Cached token is reused until it nears expiration
3. **Automatic Refresh**: Token is refreshed automatically when it reaches the buffer threshold (default: 5 minutes before expiration)
4. **Concurrent Requests**: Thread-safe operations prevent duplicate token requests during concurrent workflow execution

You can adjust the refresh buffer with the `token_cache_buffer_seconds` configuration option:

```yaml
authentication:
  my_service_account:
    _type: mcp_service_account
    # ... other configuration ...
    token_cache_buffer_seconds: 600  # Refresh 10 minutes before expiry
```

## Example Workflow

The Service Account Authentication Example, `examples/MCP/service_account_auth_mcp/configs/config-mcp-service-account-jira.yml`, provides a complete example of using service account authentication to access a protected MCP server.

See the `examples/MCP/service_account_auth_mcp/README.md` for instructions on how to run the example.

## Troubleshooting
### Error: "`client_id` is required"

Ensure the `client_id` field is set in your configuration or the corresponding environment variable is defined.

### Error: "Invalid service account credentials"

Verify your client ID and client secret are correct, the token endpoint URL is reachable, and your service account has necessary permissions.

### Error: "Service account rate limit exceeded"

Wait before retrying. Consider increasing `token_cache_buffer_seconds` to reduce token refresh frequency.

### Error: "SSL: CERTIFICATE_VERIFY_FAILED"

The MCP server uses certificates from an internal Certificate Authority. Install your organization's CA certificates in your system's trust store.

The MCP Python SDK does not currently support disabling SSL verification. See [MCP Python SDK Issue #870](https://github.com/modelcontextprotocol/python-sdk/issues/870) for updates.

### Authentication works locally but fails in CI/CD

Verify all environment variables are set in your CI/CD platform and check secret management configuration.

## See Also

- [MCP Authentication](./mcp-auth.md) - OAuth2 interactive authentication for user-facing workflows
- [Secure Token Storage](./mcp-auth-token-storage.md) - Token storage and management best practices
- [MCP Client](./mcp-client.md) - Connecting to MCP servers
