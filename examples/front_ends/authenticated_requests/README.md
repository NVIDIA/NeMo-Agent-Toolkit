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

# Authenticated Requests Example

This example demonstrates the AuthProviders extensibility in the NeMo Agent Toolkit, showing how to make authenticated HTTP requests to external APIs. The example includes a custom `authenticated_request` function that can make both authenticated and public HTTP requests using various authentication providers.

For more information about authentication providers and creating custom authentication integrations, see the [Authentication Provider Documentation](../../../docs/source/extend/adding-an-authentication-provider.md).

The example implements a custom `authenticated_request` function that can be used by agents to make HTTP requests to external APIs. See the function implementation in [`authenticated_request_function`](src/authenticated_requests/api_requests.py).

## OAuth 2.0 Setup for Jira

This example uses **OAuth 2.0 Authorization Code Flow** to authenticate with Atlassian's Jira API.

### Creating a Jira OAuth App

Before using this example, you need to create an OAuth app in your Atlassian Developer account to obtain the required client ID and secret:

1. **Create an OAuth App**: Follow the detailed instructions at [Atlassian's OAuth 2.0 3LO documentation](https://developer.atlassian.com/cloud/jira/platform/oauth-2-3lo-apps/#faq1) to create your OAuth app
2. **Configure Redirect URI**:
   - **Local Development**: Set the redirect URI to `http://localhost:8000/auth/redirect` in your OAuth app settings
   - **Production**: Update the redirect URI to match your production domain (e.g., `https://<yourdomain>.com/auth/redirect`)
3. **Required Scopes**: Ensure your app has the necessary Jira API scopes (already configured in the example's [config.yml](configs/config.yml))

### Configuration Details

**Required Environment Variables:**
- `JIRA_OAUTH_CLIENT_ID` - Your Jira OAuth app client ID (from Atlassian Developer Console)
- `JIRA_OAUTH_CLIENT_SECRET` - Your Jira OAuth app client secret (from Atlassian Developer Console)

**Pre-configured Settings in config.yml:**
- `redirect_uri`: `http://localhost:8000/auth/redirect`
- `authorization_url`: `https://auth.atlassian.com/authorize`
- `token_url`: `https://auth.atlassian.com/oauth/token`
- Required Jira API scopes are already included

## Running the Example

### Install Dependencies
```bash
uv pip install -e examples/front_ends/authenticated_requests
```

### Set Environment Variables
```bash
export JIRA_OAUTH_CLIENT_ID=your_client_id_here
export JIRA_OAUTH_CLIENT_SECRET=your_client_secret_here
```

### Start the Agent
```bash
nat serve --config_file=examples/front_ends/authenticated_requests/configs/config.yml
```

### Deploy the UI
Follow the instructions in the [NeMo Agent Toolkit UI](../../../external/nat-ui/README.md) to deploy the web interface.

### Query the Agent

Open the NeMo Agent Toolkit UI in your browser at `http://localhost:3000`. Ensure settings are configured correctly to point to your agent's API endpoint at `http://localhost:8000` and
the WebSocket URL at `ws://localhost:8000/websocket`.

Close the settings window. In your chat window, ensure that `Websocket` mode is enabled by navigating to the top-right corner and selecting the `Websocket` option in the arrow pop-out.

Once you've successfully connected to the websocket, you can start querying the agent. Asking the agent the following query should initiate the demonstrative authentication flow and then return
information about the authenticated user:

**Test prompt (authentication required):**
```
Send an authenticated GET request to https://api.atlassian.com/oauth/token/accessible-resources and return the cloud ID for my Jira authentication provider.
```

**Test prompt (no authentication required):**
```
Send a GET request to https://catfact.ninja/fact and show me the random cat fact from the response.
```
