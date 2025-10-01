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

# Secure Token Storage for MCP Authentication

The NeMo Agent toolkit provides a configurable, secure token storage mechanism for Model Context Protocol (MCP) OAuth2 authentication. You can store tokens securely using the toolkit's object store infrastructure, which provides encryption at rest, access controls, and persistence across service restarts.

## Overview

When using MCP with OAuth2 authentication, the toolkit needs to store authentication tokens for each user. The secure token storage feature provides:

- **Encryption at rest**: Tokens are stored in object stores that support encryption
- **Flexible backends**: Choose from in-memory (default), S3, MySQL, Redis, or custom object stores
- **Persistence**: Tokens persist across restarts when using external storage backends
- **Multi-user support**: Tokens are isolated per user with proper access controls
- **Automatic refresh**: Supports OAuth2 token refresh flows

## Architecture

The token storage system uses an abstraction layer that allows different storage backends:

```
MCPOAuth2Provider
    ↓
OAuth2AuthCodeFlowProvider
    ↓
TokenStorageBase (abstract)
    ↓
┌──────────────────┬────────────────────────┐
│                  │                        │
InMemoryTokenStorage  ObjectStoreTokenStorage
(default)           (S3, MySQL, Redis)
```

### Components

The token storage system includes three main components:

1. **TokenStorageBase**: Abstract interface defining `store()`, `retrieve()`, `delete()`, and `clear_all()` operations.
2. **InMemoryTokenStorage**: Default implementation using the toolkit's in-memory object store.
3. **ObjectStoreTokenStorage**: Implementation backed by configurable object stores such as S3, MySQL, and Redis.

## Configuration

### Default Configuration (In-Memory Storage)

By default, MCP OAuth2 authentication uses secure in-memory storage. No additional configuration is required:

```yaml
authentication_providers:
  mcp_auth:
    _type: mcp_oauth2
    server_url: https://mcp-server.example.com
    redirect_uri: http://localhost:8000/auth/callback
    client_id: your-client-id
    client_secret: your-client-secret
    scopes:
      - mcp.tools
      - mcp.resources
```

This configuration provides the following behavior:
- Tokens are stored in memory using the secure in-memory object store.
- Tokens are cleared when the service restarts.
- This setup is suitable for development and testing environments.

### External Object Store Configuration

For production environments, configure an external object store to persist tokens across restarts:

#### Using S3-Compatible Storage

```yaml
object_stores:
  token_store:
    _type: s3
    endpoint_url: https://s3.amazonaws.com
    access_key: ${AWS_ACCESS_KEY_ID}
    secret_key: ${AWS_SECRET_ACCESS_KEY}
    bucket_name: mcp-auth-tokens
    region: us-west-2

authentication_providers:
  mcp_auth:
    _type: mcp_oauth2
    server_url: https://mcp-server.example.com
    redirect_uri: http://localhost:8000/auth/callback
    client_id: your-client-id
    client_secret: your-client-secret
    scopes:
      - mcp.tools
      - mcp.resources
    token_storage_object_store: token_store
```

#### Using MySQL Storage

```yaml
object_stores:
  token_store:
    _type: mysql
    host: localhost
    port: 3306
    username: root
    password: ${MYSQL_PASSWORD}
    bucket_name: mcp_tokens

authentication_providers:
  mcp_auth:
    _type: mcp_oauth2
    server_url: https://mcp-server.example.com
    redirect_uri: http://localhost:8000/auth/callback
    client_id: your-client-id
    client_secret: your-client-secret
    token_storage_object_store: token_store
```

#### Using Redis Storage

```yaml
object_stores:
  token_store:
    _type: redis
    host: localhost
    port: 6379
    db: 0
    bucket_name: mcp_tokens

authentication_providers:
  mcp_auth:
    _type: mcp_oauth2
    server_url: https://mcp-server.example.com
    redirect_uri: http://localhost:8000/auth/callback
    client_id: your-client-id
    client_secret: your-client-secret
    token_storage_object_store: token_store
```

## Token Storage Format

The system stores tokens as JSON-serialized `AuthResult` objects in the object store with the following structure:

- **Key format**: `tokens/{user_id}`
- **Content type**: `application/json`
- **Metadata**: Includes token expiration timestamp when available

Example stored token:
```json
{
  "credentials": [
    {
      "kind": "bearer",
      "token": "encrypted_token_value",
      "scheme": "Bearer",
      "header_name": "Authorization"
    }
  ],
  "token_expires_at": "2025-10-02T12:00:00Z",
  "raw": {
    "access_token": "...",
    "refresh_token": "...",
    "expires_at": 1727870400
  }
}
```

## Token Lifecycle

### 1. Initial Authentication

When a user first authenticates, the system completes the following steps:
1. The OAuth2 flow completes and returns an access token.
2. The token is serialized and stored using the configured storage backend.
3. The token is associated with the user's session ID.

### 2. Token Retrieval

On subsequent requests, the system completes the following steps:
1. The user's session ID is extracted from cookies.
2. The stored token is retrieved from the storage backend.
3. The token expiration is checked.
4. If expired, a token refresh is attempted.

### 3. Token Refresh

When a token expires, the system completes the following steps:
1. The refresh token is extracted from the stored token.
2. A new access token is requested from the OAuth2 provider.
3. The new token is stored, replacing the old one.
4. The refreshed token is returned for use.

### 4. Token Deletion

You can explicitly delete tokens, or the system automatically removes them when:
- The user logs out.
- The token refresh fails.
- The storage is cleared.

## Security Considerations

### Default In-Memory Storage

The default in-memory storage provides the following characteristics:
- Tokens stored in the process memory space.
- No persistence to disk.
- Tokens are cleared on service restart.
- This option is suitable for development and single-instance deployments.

### External Object Store Storage

External storage backends provide the following capabilities:
- **Encryption at rest**: S3 supports server-side encryption, and MySQL can use encrypted connections.
- **Access controls**: Configure object stores with IAM policies and network restrictions.
- **Audit logging**: Track token access patterns.
- **Persistence**: Tokens survive service restarts.
- **High availability**: Use replicated storage for production deployments.

### Best Practices

1. **Use environment variables** for sensitive configuration:
   ```yaml
   authentication_providers:
     mcp_auth:
       client_secret: ${MCP_CLIENT_SECRET}

   object_stores:
     token_store:
       secret_key: ${AWS_SECRET_ACCESS_KEY}
   ```

2. **Enable encryption on your object store**:
   - S3: Use server-side encryption such as SSE-S3 or SSE-KMS
   - MySQL: Use SSL/TLS connections
   - Redis: Use TLS encryption and authentication

3. **Set appropriate token expiration**: Configure your OAuth2 providers to use short-lived access tokens with refresh tokens.

4. **Implement token rotation**: Regularly rotate tokens using the OAuth2 refresh flow.

5. **Monitor token usage**: Enable logging and monitoring on your object store to detect unusual access patterns.

## Custom Token Storage

You can implement custom token storage by extending the `TokenStorageBase` abstract class:

```python
from nat.plugins.mcp.auth.token_storage import TokenStorageBase
from nat.data_models.authentication import AuthResult

class CustomTokenStorage(TokenStorageBase):
    async def store(self, user_id: str, auth_result: AuthResult) -> None:
        # Custom storage logic
        pass

    async def retrieve(self, user_id: str) -> AuthResult | None:
        # Custom retrieval logic
        pass

    async def delete(self, user_id: str) -> None:
        # Custom deletion logic
        pass

    async def clear_all(self) -> None:
        # Custom clear logic
        pass
```

Then configure your custom storage in the MCP provider initialization.

## Troubleshooting

### Tokens Not Persisting

**Problem**: Tokens are lost when the service restarts.

**Solution**: Configure an external object store using `token_storage_object_store` instead of relying on the default in-memory storage.

### Token Refresh Failures

**Problem**: Tokens expire and refresh fails.

**Solution**: Complete the following steps:
- Verify that your OAuth2 provider returns refresh tokens.
- Check that the `token_url` is correctly configured.
- Ensure the refresh token is not expired.

### Object Store Connection Errors

**Problem**: Cannot connect to the configured object store.

**Solution**: Complete the following steps:
- Verify network connectivity to the object store.
- Check credentials and permissions.
- Ensure the bucket or database exists.
- Review object store logs for detailed error messages.

### Performance Issues

**Problem**: Token operations are slow.

**Solution**: Try the following approaches:
- Use Redis for lowest latency token operations.
- Configure connection pooling for MySQL.
- Use a geographically closer object store endpoint.
- Consider the in-memory option for single-instance deployments.

## Examples

### Complete Example with S3 Storage

```yaml
general:
  front_end:
    _type: fastapi
    host: 0.0.0.0
    port: 8000

object_stores:
  mcp_token_store:
    _type: s3
    endpoint_url: https://s3.amazonaws.com
    access_key: ${AWS_ACCESS_KEY_ID}
    secret_key: ${AWS_SECRET_ACCESS_KEY}
    bucket_name: my-mcp-tokens
    region: us-west-2

authentication_providers:
  mcp_oauth:
    _type: mcp_oauth2
    server_url: https://mcp-server.example.com
    redirect_uri: http://localhost:8000/auth/callback
    client_id: ${MCP_CLIENT_ID}
    client_secret: ${MCP_CLIENT_SECRET}
    scopes:
      - mcp.tools
      - mcp.resources
    use_pkce: true
    token_storage_object_store: mcp_token_store

functions:
  mcp_tools:
    _type: mcp_client
    server_url: https://mcp-server.example.com
    transport: streamable-http
    auth_provider: mcp_oauth

workflow:
  _type: react_agent
  llm: nvidia_llm
  tool_names:
    - mcp_tools
```

### Development Configuration

For development with in-memory storage:

```yaml
authentication_providers:
  mcp_oauth:
    _type: mcp_oauth2
    server_url: http://localhost:3000
    redirect_uri: http://localhost:8000/auth/callback
    enable_dynamic_registration: true
    use_pkce: true
    # No token_storage_object_store specified - uses in-memory by default
```

## Related Documentation

- [MCP Client Configuration](mcp-client.md)
- [Object Store Documentation](../../store-and-retrieve/object-store.md)
- [Authentication API Reference](../../reference/api-authentication.md)
- [Extending Object Stores](../../extend/object-store.md)

## API Reference

### Configuration Fields

#### `token_storage_object_store`
- **Type**: `ObjectStoreRef | None`
- **Default**: `None`
- **Description**: Reference to an object store for secure token storage. If `None`, uses the default in-memory storage.

### Token Storage Interface

#### `TokenStorageBase`

Abstract base class for token storage implementations.

**Methods**:

- `async def store(user_id: str, auth_result: AuthResult) -> None`
  - Store an authentication result for a user.

- `async def retrieve(user_id: str) -> AuthResult | None`
  - Retrieve an authentication result for a user.
  - Returns `None` if no token is found.

- `async def delete(user_id: str) -> None`
  - Delete an authentication result for a user.

- `async def clear_all() -> None`
  - Clear all stored authentication results.

## Summary

The secure token storage feature provides a flexible and secure way to manage OAuth2 tokens for MCP authentication.

Key features include:

- **Default**: Secure in-memory storage for development.
- **Production**: Configure S3, MySQL, or Redis for persistent, encrypted storage.
- **Extensible**: Implement custom storage backends as needed.
- **Automatic**: Token refresh and lifecycle management handled automatically.

For production deployments, always use an external object store with encryption enabled and appropriate access controls configured.
