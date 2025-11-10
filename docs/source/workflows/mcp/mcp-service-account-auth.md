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
- **Thread-Safe Operations**: Concurrent request handling with proper locking mechanisms

## Configuring Service Account Auth Provider

The `mcp_service_account` provider is a built-in authentication provider in the NVIDIA NeMo Agent toolkit. Configure it in your workflow YAML file:

```yaml
authentication:
  my_service_account:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: https://auth.example.com/oauth/token
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

:::{note}
Service account authentication is only supported with the `streamable-http` transport. The `stdio` transport does not require authentication (local process), and the `sse` transport does not support authentication.
:::

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
```
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
```
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

### Access Controls

- **Principle of least privilege**: Grant service accounts only necessary permissions
- **Audit access**: Monitor and audit service account usage
- **Separate accounts**: Use different service accounts for different services or environments

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

## Comparison: Service Account vs OAuth2 Interactive

Choose the appropriate authentication method based on your use case:

| Feature | Service Account (`mcp_service_account`) | OAuth2 Interactive (`mcp_oauth2`) |
|---------|----------------------------------------|-----------------------------------|
| User Interaction | None (headless) | Required (browser-based) |
| Use Case | Automated workflows, CI/CD, backend services | User-facing applications |
| Authentication Flow | Client credentials | Authorization code |
| Token Scope | Service-level permissions | User-level permissions |
| Session | Shared across invocations | Per-user sessions |
| Setup Complexity | Simple (client ID and secret) | Complex (redirect URIs, consent screens) |
| Best For | Automation and integration | Interactive applications with user context |

## Example Workflow

The Service Account Authentication Example, `examples/MCP/service_account_auth_mcp/README.md`, provides a complete example of using service account authentication to access a protected MCP server.

### Example Configuration

```yaml
function_groups:
  mcp_api:
    _type: mcp_client
    server:
      transport: streamable-http
      url: https://api.example.com/mcp
      auth_provider: service_account_auth

authentication:
  service_account_auth:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes: ${SERVICE_ACCOUNT_SCOPES}

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0

workflow:
  _type: react_agent
  tool_names: [mcp_api]
  llm_name: nim_llm
```

### Running the Example

```bash
# Set required environment variables
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
export SERVICE_ACCOUNT_CLIENT_SECRET="your-client-secret"
export SERVICE_ACCOUNT_TOKEN_URL="https://auth.example.com/oauth/token"
export SERVICE_ACCOUNT_SCOPES="api.read api.write"

# Run the workflow
nat run --config_file config.yml --input "Your query here"
```

The workflow will authenticate automatically without any browser interaction.

## Troubleshooting

### Common Issues and Solutions

**Error: "client_id is required"**

**Cause**: The `client_id` field is missing or not properly set.

**Solution**: Ensure the `SERVICE_ACCOUNT_CLIENT_ID` environment variable is set or provide the value directly in the configuration.

```bash
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
```

---

**Error: "Invalid service account credentials"**

**Cause**: The client ID or client secret is incorrect, or the service account does not have access to the requested endpoint.

**Solution**:
- Verify client ID and client secret are correct
- Check that the token endpoint URL is reachable
- Confirm the service account has been granted necessary permissions

---

**Error: "Service account rate limit exceeded"**

**Cause**: Too many token requests to the OAuth2 endpoint.

**Solution**:
- Wait before retrying
- Increase `token_cache_buffer_seconds` to reduce token refresh frequency
- Check for issues causing excessive token requests (for example, incorrect expiration handling)

---

**Tokens expiring unexpectedly**

**Cause**: Token cache buffer may be too aggressive, or OAuth2 server has short token lifetimes.

**Solution**:
- Adjust `token_cache_buffer_seconds` to a smaller value
- Contact your OAuth2 administrator about token lifetime policies
- Monitor token expiration times in debug logs

---

**Authentication works locally but fails in CI/CD**

**Cause**: Environment variables may not be properly configured in the CI/CD environment.

**Solution**:
- Verify all required environment variables are set in your CI/CD platform
- Check for proper secret management configuration
- Ensure the service account has necessary permissions in the target environment

### Debug Logging

Enable debug logging to troubleshoot authentication issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Look for log messages from:
- `nat.plugins.mcp.auth.service_account.provider`
- `nat.plugins.mcp.auth.service_account.token_client`

Debug logs will show:
- Token acquisition attempts
- Token cache hits and misses
- Token expiration times
- Authentication header formats

## See Also

- [MCP Authentication](./mcp-auth.md) - OAuth2 interactive authentication for user-facing workflows
- [Secure Token Storage](./mcp-auth-token-storage.md) - Token storage and management best practices
- [MCP Client](./mcp-client.md) - Connecting to MCP servers
- [Service Account Example](../../examples/MCP/service_account_auth_mcp/README.md) - Complete working example
