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

# Secure Token Storage for Model Context Protocol (MCP) Authentication

The NVIDIA NeMo Agent toolkit provides a configurable, secure token storage mechanism for Model Context Protocol (MCP) OAuth2 authentication. You can store tokens securely using the object store infrastructure, which provides encryption at rest, access controls, and persistence across service restarts.

## Overview

When using MCP with OAuth2 authentication, the toolkit needs to store authentication tokens for each user. The secure token storage feature provides:

- **Encryption at rest**: Stores tokens in object stores that support encryption
- **Flexible backends**: Allows you to choose from in-memory (default), S3, MySQL, Redis, or custom object stores
- **Persistence**: Persists tokens across restarts when using external storage backends
- **Multi-user support**: Isolates tokens per user with proper access controls
- **Automatic refresh**: Supports OAuth2 token refresh flows

### Components

The token storage system includes three main components:

1. **TokenStorageBase**: Abstract interface defining `store()`, `retrieve()`, `delete()`, and `clear_all()` operations.
2. **InMemoryTokenStorage**: Default implementation using the in-memory object store.
3. **ObjectStoreTokenStorage**: Implementation backed by configurable object stores such as S3, MySQL, and Redis.

## Configuration
This section describes the ways you can configure your token storage.

### Default Configuration (In-Memory Storage)

By default, MCP OAuth2 authentication uses in-memory storage. The following is the default configuration with no additional configuration required.
:::{note}
This setup is suitable only for development and testing environments because it uses in-memory storage that is not persistent and unsafe.
:::

```yaml
authentication:
  mcp_oauth2_jira:
    _type: mcp_oauth2
    server_url: ${CORPORATE_MCP_JIRA_URL}
    redirect_uri: http://localhost:8000/auth/redirect
    default_user_id: ${CORPORATE_MCP_JIRA_URL}
    allow_default_user_id_for_tool_calls: ${ALLOW_DEFAULT_USER_ID_FOR_TOOL_CALLS:-true}
```

### External Object Store Configuration

For production environments, configure an external object store to persist tokens across restarts. The NeMo Agent toolkit supports S3-compatible storage (for example, MinIO and AWS S3), MySQL, and Redis backends.

:::{note}
For detailed object store setup instructions including MinIO, MySQL, and Redis installation and configuration examples, refer to the `examples/object_store/user_report/README.md` guide, under the **Choose an Object Store** section.
:::

The following example shows token storage configuration using S3-compatible storage (MinIO):

```yaml
object_stores:
  token_store:
    _type: s3
    endpoint_url: http://localhost:9000
    access_key: minioadmin
    secret_key: minioadmin
    bucket_name: my-bucket

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
    token_storage_object_store: token_store

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 1024

workflow:
  _type: react_agent
  tool_names:
    - mcp_jira
  llm_name: nim_llm
  verbose: true
  retry_parsing_errors: true
  max_retries: 3
```

For MySQL or Redis configurations, replace the `object_stores` section with the appropriate object store type. Refer to the [Object Store Documentation](../../store-and-retrieve/object-store.md) for configuration options for each backend.

## Token Storage Format

The system stores tokens as JSON-serialized `AuthResult` objects in the object store with the following structure:

- **Key format**: `tokens/{sha256_hash}` where the hash is computed from the `user_id` to ensure S3 compatibility
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

### Initial Authentication

When a user first authenticates, the system completes the following steps:
1. The OAuth2 flow completes and returns an access token.
2. The token is serialized and stored using the configured storage backend.
3. The token is associated with the user's session ID.

### Token Retrieval

On subsequent requests, the system completes the following steps:
1. The user's session ID is extracted from cookies.
2. The stored token is retrieved from the storage backend.
3. The token expiration is checked.
4. If expired, a token refresh is attempted.

### Token Refresh

When a token expires, the system completes the following steps:
1. The refresh token is extracted from the stored token.
2. A new access token is requested from the OAuth2 provider.
3. The new token is stored, replacing the old one.
4. The refreshed token is returned for use.


## Custom Token Storage

Create custom token storage by extending the `TokenStorageBase` abstract class:

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


## Related Documentation

- [MCP Client Configuration](mcp-client.md)
- [Object Store Documentation](../../store-and-retrieve/object-store.md)
- [Authentication API Reference](../../reference/api-authentication.md)
- [Extending Object Stores](../../extend/object-store.md)
