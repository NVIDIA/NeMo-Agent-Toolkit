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

# MCP Service Account Authentication Example

This example demonstrates how to use the NVIDIA NeMo Agent toolkit with MCP servers that support service account authentication. Service account authentication enables headless, automated workflows without requiring browser-based user interaction.

It is recommended to read the [MCP Service Account Authentication](../../../docs/source/workflows/mcp/mcp-service-account-auth.md) documentation first.

## Overview

Service account authentication uses OAuth2 client credentials flow instead of the interactive authorization code flow. This makes it ideal for:

- **CI/CD Pipelines**: Automated testing and deployment
- **Backend Services**: Server-to-server communication
- **Batch Processing**: Scheduled jobs and data processing
- **Container Deployments**: Containerized applications
- **Any Headless Scenario**: Where browser interaction is not possible

### Authentication Patterns

This example demonstrates two service account authentication patterns:

1. **Dual Authentication (Jira example)**: Requires both an OAuth2 service account token AND a service token
   - Used by enterprise data MCP servers (such as Jira, GitLab)
   - MCP server validates the OAuth2 service account token and uses the service token (service-specific token such as Jira service token or GitLab service token) to access backend APIs
   - Two authentication headers sent with each request

2. **Single Authentication (Jama Cache example)**: Requires only an OAuth2 service account token
   - Used by custom MCP servers without service token delegation
   - MCP server validates only the OAuth2 service account token
   - Simpler authentication flow with one authentication header

## Prerequisites

1. **MCP Server Access**: Access to an MCP server that supports service account authentication (for example, corporate Jira system via MCP)

2. **Service Account Credentials**:
   - OAuth2 client ID and client secret
   - OAuth2 token endpoint URL
   - Required OAuth2 scopes
   - Optional: service-specific tokens (for example, Jira service account token)

## Install this Workflow

Install this example:

```bash
uv pip install -e examples/MCP/service_account_auth_mcp
```

## Configuration

This example includes two configuration files demonstrating different service account authentication patterns:

1. **`config-mcp-service-account-jira.yml`**: Demonstrates dual authentication (OAuth2 service account token + service token)
2. **`config-mcp-service-account-jama.yml`**: Demonstrates single authentication (OAuth2 service account token only)

Choose the configuration pattern that matches your MCP server's requirements.

### Environment Setup

#### Required Environment Variables (Both Patterns)

Set these environment variables for your OAuth2 service account:

```bash
# OAuth2 client credentials (required for both patterns)
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
export SERVICE_ACCOUNT_CLIENT_SECRET="your-client-secret"

# Service account token endpoint (required for both patterns)
export SERVICE_ACCOUNT_TOKEN_URL="https://auth.example.com/service_account/token"

# Service account scopes - space-separated (required for both patterns)
export SERVICE_ACCOUNT_SCOPES="service-account-scope-jama_cache service-account-scope-jira"
```

:::{note}
The OAuth2 service account token automatically includes the `service_account_ssa:` prefix. You no longer need to configure a `token_prefix` field.
:::

#### Pattern 1: Single Authentication (Jama Cache Example)

For custom MCP servers that only require OAuth2 service account token validation:

```bash
# MCP server URL
export CORPORATE_MCP_SERVICE_ACCOUNT_JAMA_URL="https://mcp.example.com/jama/mcp"
```

#### Pattern 2: Dual Authentication (Jira Example)

For enterprise MCP servers that require both OAuth2 service account token and service token:

```bash
# MCP server URL
export CORPORATE_MCP_SERVICE_ACCOUNT_JIRA_URL="https://mcp.example.com/jira/mcp"

# Service-specific token for accessing backend APIs (static token)
export JIRA_SERVICE_TOKEN="your-jira-service-token"

# Optional: Custom header name for service token (defaults to X-Service-Account-Token)
export SERVICE_TOKEN_HEADER="X-Service-Account-Token"
```

:::{tip}
**Advanced: Dynamic Service Token**

Instead of providing a static token via environment variable, you can configure a custom Python function to fetch the service token dynamically at runtime.

Function signature: `async def get_service_token(**kwargs) -> str | tuple[str, str]`

The function can access `AIQContext.get()` for runtime context and receive additional arguments via the `kwargs` field in the config. This is useful for enterprise environments with dynamic token management (e.g., fetching from secure vaults).

Example in config:
```yaml
service_token:
  function: "my_module.get_service_token"
  kwargs:
    vault_path: "secrets_jira"
  header: X-Service-Account-Token
```
:::

:::{important}
All environment variables here are for demonstration purposes. You must set the environment variables for your actual service account and MCP server URL.
:::

:::{warning}
Do not commit these environment variables to version control.
:::

## Run the Workflow

After setting the required environment variables, run the workflow with the appropriate configuration file:

### Single Authentication Pattern (Jama Cache)

```bash
nat run --config_file examples/MCP/service_account_auth_mcp/configs/config-mcp-service-account-jama.yml \
    --input "What Jama releases are available?"
```

### Dual Authentication Pattern (Jira - Static Token)

```bash
nat run --config_file examples/MCP/service_account_auth_mcp/configs/config-mcp-service-account-jira.yml \
    --input "What is status of jira ticket OCSW-2116?"
```

### Dual Authentication Pattern (Jira - Dynamic Function)

This example demonstrates fetching the service token dynamically via a Python function instead of reading from environment variables:

```bash
nat run --config_file examples/MCP/service_account_auth_mcp/configs/config-mcp-service-account-jira-function.yml \
    --input "What is status of jira ticket OCSW-2116?"
```

The function is defined in `scripts/service_tokens.py` and demonstrates how to implement dynamic token retrieval. In production, you would replace this with logic to fetch tokens from secure vaults or token services.

## Expected Behavior

When using service account authentication:

1. **No Browser Interaction**: The workflow runs completely headless without opening a browser
2. **Automatic Token Acquisition**: OAuth2 tokens are automatically obtained using client credentials
3. **Token Caching**: Tokens are cached and reused until they near expiration (5-minute buffer by default)
4. **Automatic Refresh**: Tokens are refreshed automatically before expiry
5. **Silent Failure Recovery**: Transient authentication errors trigger automatic retry with fresh tokens

## Troubleshooting

For common issues and solutions, see the [Troubleshooting section](../../../docs/source/workflows/mcp/mcp-service-account-auth.md#troubleshooting) in the Service Account Authentication documentation.

## Adapting This Example

### Choosing the Right Pattern

First, determine which authentication pattern your MCP server requires:

- **Use Single Authentication (Jama Cache pattern)** if your MCP server:
  - Only validates OAuth2 service account tokens
  - Does not require service tokens
  - Is a custom MCP server without backend system delegation

- **Use Dual Authentication (Jira pattern)** if your MCP server:
  - Requires both OAuth2 service account token validation and service tokens
  - Delegates access to backend systems (such as Jira, GitLab)
  - Needs dual-header authentication

### Adapting the Configuration

To use this example with your own service:

1. Choose the appropriate configuration file as your starting point
2. Update the environment variables to match your service's requirements
3. Modify the MCP server URL in the configuration file
4. For dual authentication, configure the service token header name and service token
5. Adjust the token prefix if your service uses a different format

For detailed configuration options and authentication patterns, refer to the [MCP Service Account Authentication](../../../docs/source/workflows/mcp/mcp-service-account-auth.md) documentation.

## See Also

- [MCP Service Account Authentication](../../../docs/source/workflows/mcp/mcp-service-account-auth.md) - Complete configuration reference and authentication patterns
- [MCP Authentication](../../../docs/source/workflows/mcp/mcp-auth.md) - OAuth2 interactive authentication for user-facing workflows
- [MCP Client](../../../docs/source/workflows/mcp/mcp-client.md) - MCP client configuration guide
