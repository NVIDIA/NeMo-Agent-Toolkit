# MCP Service Account Authentication

Generic OAuth2 client credentials authentication for headless MCP workflows.

## Overview

The `mcp_service_account` auth provider enables automated, headless authentication for MCP clients using OAuth2 client credentials flow. This is ideal for:

- CI/CD pipelines
- Automated workflows
- Backend services
- Batch processing
- Any scenario where interactive user authentication is not possible

## Features

- **Generic OAuth2 Support**: Works with any OAuth2-compliant token endpoint
- **Custom Token Formats**: Supports custom token prefixes for non-standard Bearer token formats
- **Multi-Header Support**: Can inject multiple headers using NAT's HeaderCred (such as Authorization and service-specific tokens)
- **Token Caching**: Automatic token caching and refresh with configurable buffer time
- **Thread-Safe**: Concurrent request handling with proper locking
- **No Hard-Coded Values**: All deployment-specific values provided via configuration or environment variables

## Configuration

### Required Fields

| Field | Environment Variable | Description |
|-------|---------------------|-------------|
| `client_id` | `SERVICE_ACCOUNT_CLIENT_ID` | OAuth2 client identifier |
| `client_secret` | `SERVICE_ACCOUNT_CLIENT_SECRET` | OAuth2 client secret |
| `token_url` | `SERVICE_ACCOUNT_TOKEN_URL` | OAuth2 token endpoint URL |
| `scopes` | `SERVICE_ACCOUNT_SCOPES` | Space-separated OAuth2 scopes |

### Optional Fields

| Field | Default | Description |
|-------|---------|-------------|
| `token_prefix` | `service_account` | Prefix for Authorization header. Use empty string for standard Bearer token. |
| `service_token` | `None` | Additional service-specific token |
| `service_token_header` | `X-Service-Token` | Header name for service token |
| `token_cache_buffer_seconds` | `300` | Seconds before expiry to refresh token |

## Usage Examples

### Basic Configuration (Single Header)

```yaml
authentication:
  my_service:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: https://auth.example.com/oauth/token
    scopes: "api.read api.write"
```

Produces header:
```
Authorization: Bearer service_account:<token>
```

### Standard Bearer Token (No Prefix)

```yaml
authentication:
  my_service:
    _type: mcp_service_account
    token_url: https://auth.example.com/oauth/token
    scopes: "api.read"
    token_prefix: ""  # Empty string = standard Bearer token
```

Produces header:
```
Authorization: Bearer <token>
```

### Two-Header Pattern (Authorization + Service Token)

```yaml
authentication:
  my_service:
    _type: mcp_service_account
    token_url: https://auth.example.com/oauth/token
    scopes: "service.scope"
    token_prefix: service_account_ssa
    service_token: ${SERVICE_ACCOUNT_SERVICE_TOKEN}
    service_token_header: NV-Service-Account-Token
```

Produces headers:
```
Authorization: Bearer service_account_ssa:<token>
NV-Service-Account-Token: <service_token>
```

### Environment-Only Configuration

```yaml
authentication:
  my_service:
    _type: mcp_service_account
    # All values loaded from environment:
    # - SERVICE_ACCOUNT_CLIENT_ID
    # - SERVICE_ACCOUNT_CLIENT_SECRET
    # - SERVICE_ACCOUNT_TOKEN_URL
    # - SERVICE_ACCOUNT_SCOPES
    # - SERVICE_ACCOUNT_SERVICE_TOKEN (optional)
```

### Complete Workflow Example

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

workflow:
  _type: react_agent
  tool_names: [mcp_api]
  llm_name: nim_llm
```

## Environment Setup

```bash
# Required
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
export SERVICE_ACCOUNT_CLIENT_SECRET="your-client-secret"
export SERVICE_ACCOUNT_TOKEN_URL="https://auth.example.com/oauth/token"
export SERVICE_ACCOUNT_SCOPES="api.read api.write"

# Optional (for two-header pattern)
export SERVICE_ACCOUNT_SERVICE_TOKEN="your-service-token"

# Run workflow
nat run --config_file config.yml --input "Your query"
```

## How It Works

### Token Acquisition Flow

1. **First Request**: Client credentials sent to OAuth2 token endpoint
2. **Token Received**: Access token cached with expiration time
3. **Header Injection**: Token formatted and injected into MCP requests
4. **Auto-Refresh**: Token refreshed automatically before expiration (5 min buffer by default)
5. **Concurrent Safety**: Thread-safe caching prevents duplicate token requests

### OAuth2 Client Credentials Flow

```
+--------+                               +---------------+
|        |--(A)- Client Authentication ->| Authorization |
| Client |                               |     Server    |
|        |<-(B)---- Access Token ---------|               |
+--------+                               +---------------+
```

The provider implements OAuth2 client credentials (RFC 6749 Section 4.4) with Basic authentication:

```http
POST /oauth/token HTTP/1.1
Host: auth.example.com
Authorization: Basic base64(client_id:client_secret)
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials&scope=api.read api.write
```

## Token Caching

Tokens are automatically cached to minimize token endpoint requests:

- **Cache Hit**: Returns cached token if still valid (with 5 min buffer)
- **Cache Miss**: Fetches new token and updates cache
- **Expiration Tracking**: Monitors token expiration from OAuth2 response
- **Thread-Safe**: Concurrent requests handled with asyncio locks

## Error Handling

The provider handles common OAuth2 errors:

| Status Code | Error | Handling |
|-------------|-------|----------|
| 200 | Success | Token cached and returned |
| 401 | Invalid credentials | RuntimeError raised |
| 429 | Rate limit | RuntimeError raised |
| Other | Server error | RuntimeError with details |
| Timeout | Network timeout | RuntimeError raised |

## Security Considerations

### Credential Storage
- **Never commit credentials** to version control
- Use environment variables or secret management systems
- Rotate credentials regularly

### Token Security
- Tokens cached in memory only (not persisted to disk)
- Tokens refreshed before expiration to minimize exposure window
- Failed requests don't expose credentials in logs

### Best Practices
1. Use minimal required scopes
2. Implement token rotation policies
3. Monitor token usage for anomalies
4. Use HTTPS for all token endpoint communications
5. Protect client secrets with appropriate access controls

## Comparison with OAuth2 Interactive Auth

| Feature | Service Account | OAuth2 Interactive |
|---------|----------------|-------------------|
| User interaction | None (headless) | Required (browser) |
| Use case | Automated workflows | User-facing applications |
| Auth flow | Client credentials | Authorization code |
| Token scope | Service-level | User-level |
| Session | Shared | Per-user |
| Setup | Simple | Complex (redirect URIs, etc.) |

## Troubleshooting

### Common Issues

**Error: "client_id is required"**
- Solution: Set `SERVICE_ACCOUNT_CLIENT_ID` environment variable or provide in config

**Error: "Invalid service account credentials"**
- Solution: Verify client_id and client_secret are correct
- Check token endpoint URL is reachable

**Error: "Service account rate limit exceeded"**
- Solution: Token endpoint rate limiting active. Wait and retry.
- Consider increasing `token_cache_buffer_seconds` to reduce requests

**Tokens expiring too quickly**
- Solution: Increase `token_cache_buffer_seconds` (default: 300s)
- Check OAuth2 server's token expiration policies

### Debug Logging

Enable debug logging to troubleshoot authentication issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Look for log messages from:
- `nat.plugins.mcp.auth.service_account.provider`
- `nat.plugins.mcp.auth.service_account.token_client`

## Architecture

### File Structure

```
auth/service_account/
├── __init__.py           # Module exports
├── provider.py           # MCPServiceAccountProvider
├── provider_config.py    # Configuration model
├── token_client.py       # OAuth2 token acquisition
└── README.md            # This file
```

### Integration with NAT Core

The service account provider integrates with NAT through:

1. **AuthResult.headers**: Custom headers injected by auth provider
2. **AuthAdapter**: Checks for custom headers before standard credentials
3. **Registration**: Auto-registered as `mcp_service_account` type

This allows seamless integration with existing NAT infrastructure while supporting non-standard authentication patterns.

## License

Apache 2.0 - See LICENSE file for details
