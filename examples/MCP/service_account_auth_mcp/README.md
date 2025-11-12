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

This example demonstrates how to use the NVIDIA NeMo Agent toolkit with MCP servers that require service account authentication. Service account authentication enables headless, automated workflows without requiring browser-based user interaction.

It is recommended to read the [MCP Service Account Authentication](../../../docs/source/workflows/mcp/mcp-service-account-auth.md) documentation first.

## Overview

Service account authentication uses OAuth2 client credentials flow instead of the interactive authorization code flow. This makes it ideal for:

- **CI/CD Pipelines**: Automated testing and deployment
- **Backend Services**: Server-to-server communication
- **Batch Processing**: Scheduled jobs and data processing
- **Container Deployments**: Containerized applications
- **Any Headless Scenario**: Where browser interaction is not possible

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

This example includes a configuration file demonstrating service account authentication with a Jira MCP server. The configuration uses dual-header authentication (OAuth2 token + service-specific token), which is one possible pattern. Your service may only require the OAuth2 token.

### Environment Setup

#### Required Environment Variables

Set these environment variables for your OAuth2 service account:

```bash
# MCP server URL
export CORPORATE_MCP_SERVICE_ACCOUNT_JIRA_URL="https://mcp.example.com/jira/mcp"

# OAuth2 client credentials
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
export SERVICE_ACCOUNT_CLIENT_SECRET="your-client-secret"

# Service account token endpoint
export SERVICE_ACCOUNT_TOKEN_URL="https://auth.example.com/service_account/token"

# Service account scopes (space-separated)
export SERVICE_ACCOUNT_SCOPES="api.read api.write"
```

:::{important}
All environment variables here are for demonstration purposes. You must set the environment variables for your actual service account and MCP server URL.
:::

#### Optional Environment Variables

For dual-header authentication patterns you may need to set additional environment variables.

```bash
# Custom token prefix for Authorization header. This is used to prefix the token in the Authorization header:
# Example: "Authorization: Bearer: service_account_myauth:<access_token>"
export AUTHORIZATION_TOKEN_PREFIX="service_account_myauth"

# Custom header name for service token. This is used to add an additional header to the request:
# Example: "MyCompany-Service-Account-Token: <your-service_token>"
export SERVICE_TOKEN_HEADER="MyCompany-Service-Account-Token"

# Service-specific token. This is used to add an additional header to the request:
# Example: "MyCompany-Service-Account-Token: <your-service_token>"
export JIRA_SERVICE_TOKEN="your-service-token"

```

:::{important}
All environment variables here are for demonstration purposes. You must set the environment variables for your actual service account.
:::

:::{warning}
Do not commit these environment variables to version control.
:::

## Run the Workflow

After setting the required environment variables, run the workflow:

```bash
nat run --config_file examples/MCP/service_account_auth_mcp/configs/config-mcp-service-account-jira.yml \
    --input "What is status of jira ticket OCSW-2116?"
```

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

To use this example with your own service:

1. Update the environment variables to match your service's requirements
2. Modify the MCP server URL in the configuration file
3. Adjust authentication headers if your service uses different patterns (see the [Authentication Patterns](../../../docs/source/workflows/mcp/mcp-service-account-auth.md#authentication-patterns) section)
4. Remove optional environment variables if your service only requires OAuth2 Bearer token

For detailed configuration options and authentication patterns, refer to the [MCP Service Account Authentication](../../../docs/source/workflows/mcp/mcp-service-account-auth.md) documentation.

## See Also

- [MCP Service Account Authentication](../../../docs/source/workflows/mcp/mcp-service-account-auth.md) - Complete configuration reference and authentication patterns
- [MCP Authentication](../../../docs/source/workflows/mcp/mcp-auth.md) - OAuth2 interactive authentication for user-facing workflows
- [MCP Client](../../../docs/source/workflows/mcp/mcp-client.md) - MCP client configuration guide
