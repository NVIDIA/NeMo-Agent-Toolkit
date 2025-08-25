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

# Using Authentication in the NeMo Agent Toolkit

This example demonstrates how to use the library's native support for authentication to allow agents to use tools that require
authentication to use. Particularly, this example highlights how to use the `OAuth 2.0 Authorization Code Flow` to authenticate
with a demonstrative `OAuth 2.0` provider and then return information from the authorization server's demonstrative `/api/me` endpoint
which provides information about the authenticated user.

## Installation

First, install the `simple_auth` example:

```bash
uv pip install -e examples/front_ends/simple_auth
```

## How the OAuth2.0 Authorization‑Code Flow Works

1. **Agent launches login** – it sends the user’s browser to the OAuth provider’s
   `GET /oauth/authorize` endpoint with parameters:
   `client_id`, `redirect_uri`, requested `scope`, and a random `state`.
2. **User authenticates & grants consent** on the provider’s UI.
3. **Provider redirects back** to `redirect_uri?code=XYZ&state=…` on your app.
4. **Agent exchanges the code** for tokens by POST‑ing to `POST /oauth/token`
   with the **authorization code**, its `client_id`, the **client secret** (or PKCE
   verifier for public clients), and the same `redirect_uri`.
5. The provider returns a **JSON** payload:

   ```json
   {
     "access_token": "…",
     "token_type":   "Bearer",
     "expires_in":   3600,
     "refresh_token": "…",          // if scope included offline_access
     "id_token":      "…"           // if scope contained openid
   }
   ```

6. The agent stores the tokens and uses the `access_token` in the
   `Authorization: Bearer …` header when invoking tools that need auth.

*Why this flow?*

- Supports **confidential clients** (can keep a secret) *and* public clients with **PKCE**.
- Refresh tokens keep long‑running agents from re‑prompting the user.
- Works across browsers, CLI apps, and UI front‑ends.

## Running the Demo OAuth Provider Locally

In a separate terminal, you can run a demo OAuth 2.0 provider using the [`Authlib`](https://docs.authlib.org/en/latest/)
library. This will allow you to test the OAuth 2.0 Authorization Code Flow with your agent.

### Quick Start with Docker

The easiest way to get started is using Docker, which works seamlessly across all systems (macOS, Windows, Linux):

**Run the example (background mode)**
```bash
docker compose -f examples/front_ends/simple_auth/docker-compose.yml --project-directory examples/front_ends/simple_auth up -d
```

This will automatically:

- Clone the OAuth2 server example
- Install all dependencies
- Start the server on `http://localhost:5001`
- Set the necessary environment variables for local development

**Note**: The `AUTHLIB_INSECURE_TRANSPORT=1` environment variable is set automatically for local development to allow `http://` callback URLs. This should never be used in production.

Browse to **`http://localhost:5001/`** – you should see the demo home page. Sign up with any name.

**To stop the Docker services:**

```bash
docker compose -f examples/front_ends/simple_auth/docker-compose.yml --project-directory examples/front_ends/simple_auth down
```

**To stop and remove all data:**

```bash
docker compose -f examples/front_ends/simple_auth/docker-compose.yml --project-directory examples/front_ends/simple_auth down -v
```

Browse to **`http://localhost:5001/`** – you should see the demo home page. Sign up with any name.

## Registering a Dummy Client (“test”)

1. Open **Clients → Create New Client** in the demo UI.
2. Fill the form exactly as below and click **Submit**:

| Field                      | Value                                                 |
|----------------------------|-------------------------------------------------------|
| Client Name                | `test`                                                |
| Client URI                 | `https://test.com`                                    |
| Redirect URIs              | `http://localhost:8000/auth/redirect`                 |
| Allowed Grant Types        | `authorization_code` and `refresh_token` on new lines |
| Allowed Response Types     | `code`                                                |
| Allowed Scope              | `openid profile email`                                |
| Token Endpoint Auth Method | `client_secret_post`                                  |

3. Copy the generated **Client ID** and **Client Secret** – you’ll need them in your agent’s config.

## Deploy the NeMo Agent Toolkit UI

Follow the instructions at the GitHub repository to deploy the [NeMo Agent Toolkit UI](https://github.com/NVIDIA/NeMo-Agent-Toolkit-UI)
to deploy the UI that works with the agent in this example. Configure it according to the instructions in the README.

## Update Your Environment Variables

Export your saved client ID and secret to the following environment variables:

```bash
export NAT_OAUTH_CLIENT_ID=<your_client_id>
export NAT_OAUTH_CLIENT_SECRET=<your_client_secret>
```

## Serve The Agent

In a new terminal, serve the agent using the following command:

```bash
nat serve --config_file=examples/front_ends/simple_auth/configs/config.yml
```

This will start a FastAPI server on `http://localhost:8000` that listens for requests from the UI and
handles authentication.

## Query the Agent

Open the NeMo Agent Toolkit UI in your browser at `http://localhost:3000`. Ensure settings are configured correctly to point to your agent's API endpoint at `http://localhost:8000` and
the WebSocket URL at `ws://localhost:8000/websocket`.

Close the settings window. In your chat window, ensure that `Websocket` mode is enabled by navigating to the top-right corner and selecting the `Websocket` option in the arrow pop-out.

Once you've successfully connected to the websocket, you can start querying the agent. Asking the agent the following query should initiate the demonstrative authentication flow and then return
information about the authenticated user:

```text
Who am I logged in as?
```

**Tip**: Remember to enable pop-ups in your browser to allow the OAuth 2.0 provider to open a new window for authentication.
