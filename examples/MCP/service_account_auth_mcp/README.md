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
# OAuth2 client credentials
export SERVICE_ACCOUNT_CLIENT_ID="your-client-id"
export SERVICE_ACCOUNT_CLIENT_SECRET="your-client-secret"

# OAuth2 token endpoint
export SERVICE_ACCOUNT_TOKEN_URL="https://auth.example.com/oauth2/token"

# OAuth2 scopes (space-separated)
export SERVICE_ACCOUNT_SCOPES="api.read api.write"

# MCP server URL
export JIRA_MCP_URL="https://mcp.example.com/jira/mcp"
```
:::{warning}
All environment variables here are for demonstration purposes. You must set the environment variables for your actual service account.
:::

#### Optional Environment Variables

For dual-header authentication patterns you may need to set additional environment variables.

```bash
# Service-specific token (optional, only if your service requires it)
export JIRA_SERVICE_TOKEN="your-service-token"

# Custom header name for service token (optional)
export JIRA_SERVICE_TOKEN_HEADER="MyCompany-Service-Account-Token"

# Custom token prefix for Authorization header (optional)
export SERVICE_ACCOUNT_TOKEN_PREFIX="service_account_myauth"
```

:::{warning}
All environment variables here are for demonstration purposes. You must set the environment variables for your actual service account.
:::

## Run the Workflow

After setting the required environment variables, run the workflow:

```bash
nat run --config_file examples/MCP/service_account_auth_mcp/configs/config-mcp-service-account-jira.yml \
    --input "What is status of ticket AIQ-1935?"
```

## Expected Behavior

When using service account authentication:

1. **No Browser Interaction**: The workflow runs completely headless without opening a browser
2. **Automatic Token Acquisition**: OAuth2 tokens are automatically obtained using client credentials
3. **Token Caching**: Tokens are cached and reused until they near expiration (5-minute buffer by default)
4. **Automatic Refresh**: Tokens are refreshed automatically before expiry
5. **Silent Failure Recovery**: Transient authentication errors trigger automatic retry with fresh tokens

## Troubleshooting

### Common Issues

**Error: "client_id is required"**

- **Solution**: Ensure `SERVICE_ACCOUNT_CLIENT_ID` environment variable is set
- Check for typos in environment variable names

**Error: "Invalid service account credentials"**

- **Solution**: Verify your client ID and client secret are correct
- Ensure the token endpoint URL is reachable
- Check that your service account has been granted necessary permissions

**Error: "Service account rate limit exceeded"**

- **Solution**: Token endpoint rate limiting is active. Wait before retrying.
- Consider increasing `token_cache_buffer_seconds` in the config to reduce token requests

**Authentication works locally but fails in CI/CD**

- **Solution**: Verify all environment variables are properly set in your CI/CD platform
- Check secret management configuration in your CI/CD system
- Ensure network access to the token endpoint from CI/CD environment
service:

### Single-Header Authentication

If your service only needs a standard OAuth2 Bearer token:

```yaml
authentication:
  my_service:
    _type: mcp_service_account
    client_id: ${SERVICE_ACCOUNT_CLIENT_ID}
    client_secret: ${SERVICE_ACCOUNT_CLIENT_SECRET}
    token_url: ${SERVICE_ACCOUNT_TOKEN_URL}
    scopes: ${SERVICE_ACCOUNT_SCOPES}
    token_prefix: ""  # Empty string for standard Bearer token
```
