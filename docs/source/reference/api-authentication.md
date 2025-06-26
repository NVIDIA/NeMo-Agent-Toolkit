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

# NVIDIA Agent Intelligence Toolkit Streamlining API Authentication
AIQ toolkit enables users to invoke API calls and retrieve data from an API resource by using a utility function that
constructs and sends HTTP requests with optional authentication. When authentication to API providers is required, it is
handled automatically on the user’s behalf, abstracting away the need for manual credential management and authentication.
Currently supported authentication plugins include OAuth 2.0 Authorization Code Flow and API keys. User authentication
credentials can be provided via the Workflow Configuration YAML file, allowing for centralized management and
configuration. The toolkit is only responsible for managing user credentials during workflow execution. Credentials are
loaded at runtime and accessed by the name specified in the YAML configuration file; they are not logged during
application execution or saved after workflow termination.

## API Configuration and Usage Walkthrough
This guide provides a step-by-step walkthrough for configuring authentication credentials and making requests to
external API providers.

## 1. Register Agent Intelligence Toolkit API Server as OAuth2.0 Client
To authenticate with a third-party API using OAuth 2.0, you must first register the application as a client with that
API provider. The Agent Intelligence Toolkit API server doubles as an OAuth 2.0 client, meaning it can be registered
with an API provider to initiate the OAuth flow, manage token exchanges, and serve requests through its exposed
endpoints. This section outlines a general approach for registering the API server as an OAuth 2.0 client with your API
provider in order to enable authentication using OAuth 2.0. While this guide outlines the general steps involved, the
exact registration process may vary depending on the provider. Please refer to the specific documentation for your API
provider to complete the registration according to their requirements.


### Access the API Provider’s Developer Console to Register the Application
Navigate to the API provider’s developer console and follow the instructions to register the API server as an authorized
application. During registration, you typically provide the following:

| **Field**           | **Description**                                                                 |
|---------------------|----------------------------------------------------------------------------------|
| **Application Name**  | A human-readable name for your application. This is shown to users during consent.|
| **Redirect URI(s)**   | The URL(s) where the API will redirect users after authorization.               |
| **Grant Type(s)**     | The OAuth 2.0 flows the toolkit supports (e.g., Authorization Code, Client Credential).         |
| **Scopes**            | The permissions your app is requesting (e.g., `read:user`, `write:data`).       |

### Registering Redirect URIs for Development vs. Production Environments
**IMPORTANT**: Most OAuth providers require exact matches for redirect URIs.

| **Environment** | **Redirect URI Format**               |  **Notes**                         |
|-----------------|---------------------------------------|------------------------------------|
| Development     | `http://localhost:8000/auth/redirect` | Often used when testing locally.   |
| Production      | `https://yourdomain.com/auth/redirect`| Should use HTTPS and match exactly.|

### Configuring Registered App Credentials in Workflow Configuration YAML
After registering your application upload your credentials in the Workflow Configuration YAML file under the
`authentication` key.

#### Common Authorization Code Grant Flow Credentials

| **Field**                | **Description**     |
|--------------------------|---------------------|
| `client_id`           | The unique ID assigned to your application by the API provider.|
| `client_secret`       | A secret key used to authenticate the application with the token server.|
| `authorization_url`   | URL to which users are redirected to authorize the application.|
| `token_url`           | URL where the authorization code is exchanged for access and optionally refresh tokens.|
| `redirect_uri`        | The callback URL the provider redirects to after user authorization. Must match the registered value.|
| `scope`               | A list of requested permissions or access scopes (e.g., `read`, `write`, `email`).|
| `state`               | An optional CSRF protection value; helps maintain request integrity between client and server.|
| `code_challenge`      | Used for PKCE; a hashed version of `code_verifier` sent in the initial authorization request.|
| `code_verifier`       | Used for PKCE; a high-entropy string used to validate the code exchange.|

#### Common Client Credentials Flow Credentials

| **Field**                | **Description**     |
|--------------------------|---------------------|
| `client_id`          | The unique ID assigned to your application by the API provider.  |
| `client_secret`      | A secret key used to authenticate the application with the token server. |
| `token_url`          | URL where the authorization code is exchanged for access and optionally refresh tokens. |
| `scope` (optional)   | A list of requested permissions or access scopes (e.g., `read`, `write`, `email`).|

#### Common Device Authorization Flow Credentials

| **Field**                  | **Description**      |
|----------------------------|----------------------|
| `client_id`                | The unique ID of your app registered with the API provider.|
| `device_authorization_url` | The endpoint where the app requests a `device_code` and `user_code`.|
| `token_url`                | The endpoint to poll and exchange the `device_code` for an access token.|
| `scope`                    | A list of requested permissions or access scopes (e.g., `read`, `write`, `email`).|
| `verification_uri`         | URL the user visits on a separate device to authorize the app.|


## 2. Configuring Authentication Credentials
In the Workflow Configuration YAML file, user credentials required for API authentication are configured under the
`authentication` key. Users should provide all required and valid credentials for each authentication method to ensure the
library can authenticate requests without encountering credential related errors. Examples of currently supported API
configurations are
[OAuth 2.0 Authorization Code Grant Flow Configuration](../../../src/aiq/authentication/api_key/api_key_config.py) and
[API Key Configuration](../../../src/aiq/authentication/oauth2/auth_code_grant_config.py).

### Authentication YAML Configuration Example
```yaml
authentication:
  example_provider_name_oauth:
    _type: oauth2_authorization_code_grant
    consent_prompt_mode: browser | frontend
    consent_prompt_key: user_consent_prompt_key
    client_server_url: user_client_server_url
    authorization_url: user_authorization_url
    authorization_token_url: user_authorization_token_url
    client_id: user_client_id
    client_secret: user_client_secret
    audience: user_audience
    scope:
      - read
      - write
  example_provider_name_api_key:
    _type: api_key
    api_key: user_api_key
    header_name: accepted_api_header_name
    header_prefix: accepted_api_header_prefix
```

### OAuth2.0 Authorization Code Grant Configuration Reference
| Field Name               | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `example_provider_name_oauth`    | A unique name used to identify the client credentials required to access the API provider.|
| `_type`                  | Specifies the authentication type. For OAuth 2.0 Authorization Code Grant authentication, set this to `oauth2_authorization_code_grant`.|
| `consent_prompt_mode`    | Specifies how the OAuth 2.0 client handles the consent prompt redirect.<br>•`browser` – Opens the system's default browser for user login.<br>•`frontend` – Redirects the consent prompt to the browser that initiated the POST request. A corresponding `consent_prompt_key` is required to receive the consent prompt redirect for the intended authentication provider.|
| `consent_prompt_key`       | A unique key used to retrieve the consent prompt redirect for the provider requesting authentication. Upon successful validation, the consent prompt redirect is returned to continue the OAuth 2.0 flow only if `frontend` is selected as the `consent_prompt_mode`.|
| `client_server_url`        | URL of the OAuth 2.0 client server.|
| `authorization_url`        | URL used to initiate the authorization flow, where an authorization code is obtained to be later exchanged for an access token.|
| `authorization_token_url`  | URL used to exchange an authorization code for an access token and optional refresh token.|
| `client_id`                | The Identifier provided when registering the OAuth 2.0 client server with an API provider.|
| `client_secret`            | A confidential string provided when registering the OAuth 2.0 client server with an API provider.|
| `audience`                 | Represents the specific API provider for which the authorization tokens were issued (provider-specific).|
| `scope`                    | List of permissions to the API provider (e.g., `read`, `write`).|

### API Key Configuration Reference
| Field Name               | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `example_provider_name_api_key`| A unique name used to identify the client credentials required to access the API provider.|
| `_type`                  | Specifies the authentication type. For API Key authentication, set this to `api_key`.|
| `api_key`                | API key value for authenticating requests to the API provider.|
| `header_name`            | The HTTP header used to transmit the API key for authenticating requests.|
| `header_prefix`          | Optional prefix for the HTTP header used to transmit the API key in authenticated requests (e.g., Bearer).|


## 3. Using API Request Utility Function
Make an authenticated or non authenticated request by invoking the `make_api_request` function within
workflow tools by acquiring an instance of the `AIQUserInteractionManager`. If no authentication is needed during the
request omit the `authentication_config_name`.

### API usage example
```python
@register_function(config_type=GithubUpdateIssueToolConfig)
async def update_github_issue_async(config: GithubUpdateIssueToolConfig, builder: Builder):
    """
    Updates an issue in a GitHub repository asynchronously.
    """
    async def _github_update_issue(issues) -> list:

      import httpx
      from aiq.builder.context import AIQContext
      aiq_context = AIQContext.get()
      user_input_manager = aiq_context.user_interaction_manager

      response: httpx.Response | None = await user_input_manager.make_api_request(
          url=f"https://api.github.com/repos/{config.repo_name}/issues",
          http_method="GET",
          authentication_config_name="github_oauth",
          headers={"Accept": "application/vnd.github+json"},
          query_params=None,
          body_data=None)

    yield FunctionInfo.from_fn(_github_update_issue,
                               description=(f"Updates a GitHub issue in the "
                                            f"repo named {config.repo_name}"),
                               input_schema=GithubUpdateIssueModelList)
```

## 4. Authentication by Application Configuration
API key authentication is supported uniformly across all application configurations and deployment modes without
requiring special handling. In contrast, OAuth2.0 authentication, particularly the authorization redirect that initiates
the `consent flow` varies depending on how the application is deployed and which components are present. In some
configurations, the browser directly handles the redirect, while in others, the front-end UI is responsible for
retrieving and managing the consent flow.

The sections below detail how OAuth2.0 authentication is handled in each supported configuration.

### OAuth2.0 Consent Prompt Behavior by Application Configuration Summary

| # | UI / API Configuration             | Deployment Configuration | Consent Prompt Handler   | Chat Completion Method |
|---|------------------------------------|---------------------|-------------------------------|-----------------------|
| 1 | AIQ Front-End UI + AIQ API Server  | Single-Host         | Front-End UI, System Browser  | HTTP, WebSocket       |
| 2 | AIQ Front-End UI + AIQ API Server  | Multi-Host          | Front-End UI                  | HTTP, WebSocket       |
| 3 | Custom UI + AIQ API Server         | Multi-Host          | Front-End UI                  | HTTP, WebSocket       |
| 4 | Headless AIQ API Server            | Single-Host         | Front-End UI, System Browser  | HTTP, WebSocket       |
| 5 | Headless AIQ API Server            | Multi-Host          | Front-End UI                  | HTTP, WebSocket       |

> ⚠️ **Important:**
> When using the `aiq serve` command, you must configure CORS settings to allow the front-end to communicate with the
> server for OAuth2.0 authentication. For example:
>
> ```yaml
> general:
>  use_uvloop: true
>  front_end:
>    _type: fastapi
>    cors:
>      allow_origins: ["http://localhost:3000"]
>      allow_headers: ["*"]
>      allow_methods: ["*"]
> ```
>
> - `allow_origins`: Must include the origin of the front-end (e.g., `http://localhost:3000`) so the browser is
> permitted to make cross-origin requests.
> - `allow_headers`: Should allow headers sent by your front-end, such as `Content-Type`, `Authorization`, etc.,
> enabling the preflight check to pass.
> - `allow_methods`: Must include `POST` because the front-end retrieves the consent flow redirect by sending a POST
> request to the `prompt-uri` endpoint, passing the `consent_prompt_key` in the request body. Without this, the browser
> will block the request.
>
> The CORS configuration is done on the users behalf when using the `aiq run` command. The API server provides
> interactive Swagger documentation at the `/docs` endpoint. This interface allows you to explore all available
>  endpoints, view required parameters, and see example request and response payloads.

### Consent Prompt Behavior Configuration Details

### 1. AIQ Front-End UI + API Server (Single-Host)
AIQ Front-End UI hosted on the same machine as the AIQ API Server.

- When a tool is invoked using the `make_api_request` function via an `HTTP` request, and the OAuth
provider is configured with the `frontend` option, the user can complete the authorization consent prompt redirect
by navigating to the `/aiq-auth` front-end UI page. There, the user will enter the `consent_prompt_key` into the modal
associated with the registered provider. A `POST` request to the `prompt-uri` endpoint will be made on their behalf to
retrieve the authorization redirect, allowing the user to log in with their credentials and complete the OAuth flow.

- When a tool is invoked using the `make_api_request` function via a `WebSocket` message, and the OAuth provider is
configured with the `frontend` option, a notification prompt will appear on the AIQ front-end UI instructing the user
to navigate to the `/aiq-auth` page to complete the authorization consent prompt redirect. There, the user will enter the
`consent_prompt_key` into the modal associated with the registered provider. A `POST` request to the `prompt-uri`
endpoint will be made on their behalf to retrieve the authorization redirect, allowing the user to log in with their
credentials and complete the OAuth flow.

- When a tool is invoked using the `make_api_request` function via an `HTTP` request or `WebSocket` message, and the
  OAuth2.0 provider is configured with the `browser` option, the user's default browser will be opened (if available)
  to complete the OAuth flow.

---

### 2. AIQ Front-End UI + API Server (Multi-Host)
AIQ Front-End UI hosted on a separate machine from the AIQ API server.

- When a tool is invoked using the `make_api_request` function via an `HTTP` request or a `WebSocket` message and the
  OAuth provider is configured with the `frontend` option. The OAuth flow follows the same steps as described above in
  the section titled `1. AIQ Front-End UI + API Server (Single-Host)`.


---
### 3. Custom UI + AIQ API Server (Multi-Host)
Custom Front-End UI hosted on a separate machine from the AIQ API server.

- When a tool is called with the `make_api_request` function initiated by a `HTTP` request. A server console log will
  prompt the user to send a `POST` request to the `prompt-uri` endpoint with the registered `consent_prompt_key` in the
  request body to retrieve the authorization redirect, allowing the user to log in with their credentials and complete the
  OAuth flow. The user is responsible for implementing the logic to send and handle the authorization redirect to
  continue the OAuth flow.

- When a tool is called with the `make_api_request` function initiated by a `Websocket` message. A server console log
  will prompt the user to send a `POST` request to the `prompt-uri` endpoint with the registered `consent_prompt_key` in
  the request body to retrieve the authorization redirect, allowing the user to log in with their credentials and complete
  the OAuth flow. The user is responsible for implementing the logic to send and handle the authorization redirect to
  continue the OAuth flow as well as the {py:mod}`aiq.data_models.interactive.HumanPromptNotification`
  [Interactive WebSocket Message](websockets.md#user-interaction-message---openai-compatible) to receive a
  notification prompt.
