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

The AIQ Toolkit simplifies API authentication by streamlining credential management and validation, enabling secure
access to API providers across a variety of runtime environments. This functionality allows users to authenticate with
protected API resources directly from workflow tools, abstracting away low-level authentication logic and enabling
greater focus on data retrieval and processing. Users can define multiple credential entries in the Workflow
Configuration YAML file, each uniquely identified by a provider name. The toolkit provides utility functions such as
`authenticate_oauth_client` to complete the authentication process, particularly for flows that require user consent,
like the OAuth 2.0 Authorization Code Flow. Authentication is supported in headless and server modes, with optional
support for consent prompts handled via a custom front-end UI. Credentials are securely loaded into memory at runtime,
accessed by provider name, and are never logged or persisted. They are available only during workflow execution to
ensure secure and centralized handling. Currently supported authentication configurations include OAuth 2.0
Authorization Code Grant Flow and API keys, each managed by dedicated authentication clients. The system is designed
for extensibility, allowing developers to introduce new credential types and clients to support additional
authentication methods and protected API access patterns.

## API Authentication Configuration and Usage Walkthrough
This guide provides a step-by-step walkthrough for configuring authentication credentials and using authentication
clients to securely authenticate and send requests to external API providers.

## 1. Register Agent Intelligence Toolkit API Server as OAuth2.0 Client
To authenticate with a third-party API using OAuth 2.0, you must first register the application as a client with that
API provider. The Agent Intelligence Toolkit API server functions as both a full featured API server and an OAuth 2.0
client. In addition to serving application specific endpoints, it can be registered with external API providers to
perform delegated access, manage tokens throughout their lifecycle, and support consent prompt handling through a custom
front end. This section outlines a general approach for registering the API server as an OAuth 2.0 client with your API
provider in order to enable delegated access using OAuth 2.0. While this guide outlines the general steps involved, the
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
After registering your application upload your API authentication credentials in the Workflow Configuration YAML file
under the `authentication` key.

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
`authentication` key. Users should provide all required and valid credentials for each authentication method to ensure
the library can authenticate requests without encountering credential related errors. Examples of currently supported
API configurations are
[OAuth 2.0 Authorization Code Grant Flow Configuration](../../../src/aiq/authentication/api_key/api_key_config.py) and
[API Key Configuration](../../../src/aiq/authentication/oauth2/auth_code_grant_config.py).

### Authentication YAML Configuration Example
```yaml
authentication:
  example_provider_name_oauth:
    _type: oauth2_authorization_code_grant
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
    raw_key: user_api_key
    header_name: accepted_api_header_name
    header_prefix: accepted_api_header_prefix
```

### OAuth2.0 Authorization Code Grant Configuration Reference
| Field Name               | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `example_provider_name_oauth`    | A unique name used to identify the client credentials required to access the API provider.|
| `_type`                  | Specifies the authentication type. For OAuth 2.0 Authorization Code Grant authentication, set this to `oauth2_authorization_code_grant`.|
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
| `raw_key`                | API key value for authenticating requests to the API provider.|
| `header_name`            | The HTTP header used to transmit the API key for authenticating requests.|
| `header_prefix`          | Optional prefix for the HTTP header used to transmit the API key in authenticated requests (e.g., Bearer).|


## 3. Using the OAuth Authentication Utility Function
The `authenticate_oauth_client` function can be used to authenticate to API resource that requires an OAuth 2.0 consent
prompt.

### Consent Prompt Authentication From Workflow Function Example
```python
class OAuth2BrowserAuthTool(FunctionBaseConfig, name="oauth2_browser_auth_tool"):
    """Authenticate to any registered API provider using OAuth2 authorization flow with browser consent handling."""
    pass


@register_function(config_type=OAuth2BrowserAuthTool)
async def oauth2_browser_auth_tool(config: OAuth2BrowserAuthTool, builder: Builder):
    """
    Authenticates to any registered API provider using OAuth 2.0 authentication code flow.

    Extracts the provider name from user prompts (e.g., "authenticate to my registered API provider: jira"),
    and uses that name to retrieve a registered authentication client. A user authentication callback is then invoked
    to initiate the OAuth 2.0 authentication flow, opening a browser to complete the consent prompt.

    All authentication credentials are stored and managed internally. The authentication client then verifies
    the authenticated connection as a TEST in this tool by making an HTTP request and handling the response.
    """

    async def _arun(authentication_provider_name: str) -> str:

        # Get the user interaction manager from context
        aiq_context = AIQContext.get()
        user_input_manager = aiq_context.user_interaction_manager

        try:
            # Get the oauth registered authentication client
            oauth_client: OAuthClientBase = await builder.get_authentication(authentication_provider_name)

            # Authenticate by calling the authenticate_oauth_client to with browser consent handling.
            authentication_error: AuthenticationError | None = await user_input_manager.authenticate_oauth_client(
                oauth_client, ConsentPromptMode.BROWSER)

            # If an authentication error occurs, the authentication flow has failed
            if authentication_error or not await oauth_client.validate_credentials():
                return (f"Failed to authenticate provider: {authentication_provider_name}: "
                        f"Error: {authentication_error.error_code if authentication_error else 'Invalid credentials'} ")

            # Make a test API call to the API provider.
            test_api_call_result: typing.Any | None = await _test_jira_api_call(oauth_client)

            return (f"Your registered API Provider name: [{authentication_provider_name}] is now authenticated.\n"
                    f"Test API Response to API Provider: {test_api_call_result}. \n")

        except Exception as e:
            logger.exception("OAuth authentication failed", exc_info=True)
            return f"OAuth authentication to '{authentication_provider_name}' failed: {str(e)}"

    yield FunctionInfo.from_fn(
        _arun,
        description=(
            "Authenticates to any registered API provider using OAuth 2.0 flow. "
            "When user mentions 'registered API provider: <name>', extract the provider name (e.g., 'jira') "
            "and pass it as authentication_provider_name parameter. Opens browser for OAuth consent and tests."))

```

## 4. Authentication by Application Configuration
Authentication methods not needing consent prompts, such as API Keys is supported uniformly across all application
configurations and deployment modes without requiring special handling. In contrast, OAuth 2.0 authentication,
especially methods that require consent prompts, can vary depending on the application's deployment and available
components. In some configurations, the system’s default browser handles the redirect directly, while in others, the
front-end UI is responsible for rendering the consent prompt in the browser.

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

- When a tool is invoked via an `HTTP` request using the `authenticate_oauth_client` function with the
`consent_prompt_handle` function parameter set to `frontend`, the user can complete the authentication consent prompt by
navigating to the `/aiq-auth` page in the AIQ front-end UI. There, the user will enter the `consent_prompt_key` into the
modal associated with the registered provider. A `POST` request to the `prompt-uri` endpoint will be made on their
behalf to retrieve the consent prompt redirect, allowing the user to log in with their credentials and complete the
OAuth flow.

- When a tool is invoked using the `authenticate_oauth_client` function with the `consent_prompt_handle` function
parameter set to `frontend` via a `WebSocket` message, a notification prompt will appear in the AIQ front-end UI,
instructing the user to navigate to the `/aiq-auth` page to complete the authentication consent prompt. There, the user
will enter the `consent_prompt_key` into the modal associated with the registered provider. A `POST` request to the
`prompt-uri` endpoint will be made on their behalf to retrieve the consent prompt redirect, allowing the user to log in
with their credentials and complete the OAuth flow.


- When a tool is invoked using the `authenticate_oauth_client` function via an `HTTP` request or `WebSocket` message
with the `consent_prompt_handle` function parameter set to `browser`, the user's default browser will be opened
(if available) to complete the OAuth flow.

---

### 2. AIQ Front-End UI + API Server (Multi-Host)
AIQ Front-End UI hosted on a separate machine from the AIQ API server.

- When a tool is invoked using the `authenticate_oauth_client` function via an `HTTP` request or `WebSocket` message
with the `consent_prompt_handle` function parameter set to `frontend`. The OAuth flow follows the same steps as
described above in the section titled `1. AIQ Front-End UI + API Server (Single-Host)`.

---
### 3. Custom UI + AIQ API Server (Multi-Host)
Custom Front-End UI hosted on a separate machine from the AIQ API server.

- When a tool is called with the `authenticate_oauth_client` function initiated by a `HTTP` request. A server console
log will prompt the user to send a `POST` request to the `prompt-uri` endpoint with the registered `consent_prompt_key`
in the request body to retrieve the authentication consent prompt redirect, allowing the user to log in with their
credentials and complete the OAuth flow. The user is responsible for implementing the logic to send and handle the
authentication redirect to continue the OAuth flow.

- When a tool is called with the `authenticate_oauth_client` function initiated by a `Websocket` message. A server
console log will prompt the user to send a `POST` request to the `prompt-uri` endpoint with the registered
`consent_prompt_key` in the request body to retrieve the authentication consent prompt redirect, allowing the user to
log in with their credentials and complete the OAuth flow. The user is responsible for implementing the logic to send
and handle the authentication redirect to continue the OAuth flow as well as the
{py:mod}`aiq.data_models.interactive._HumanPromptOAuthConsent`
[Interactive WebSocket Message](websockets.md#user-interaction-message---openai-compatible) to receive a notification
prompt.
