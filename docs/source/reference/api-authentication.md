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

# NVIDIA NeMo Agent Toolkit Streamlining API Authentication

The NeMo Agent toolkit simplifies API authentication by streamlining credential management and validation, enabling secure
access to API providers across a variety of runtime environments. This functionality allows users to authenticate with
protected API resources directly from workflow tools, abstracting away low-level authentication logic and enabling
greater focus on data retrieval and processing. Users can define multiple authentication providers in their workflow
configuration file, each uniquely identified by a provider name. The toolkit provides utility functions such as
`authenticate_oauth_client` to complete the authentication process, particularly for flows that require user consent,
like the OAuth 2.0 Authorization Code Flow. Authentication is supported in headless and server modes. Credentials are
securely loaded into memory at runtime,
accessed by provider name, and are never logged or persisted. They are available only during workflow execution to
ensure secure and centralized handling. Currently supported authentication configurations include OAuth 2.0
Authorization Code Grant Flow and API keys, each managed by dedicated authentication clients. The system is designed
for extensibility, allowing developers to introduce new credential types and clients to support additional
authentication methods and protected API access patterns.

## API Authentication Configuration and Usage Walkthrough
This guide provides a step-by-step walkthrough for configuring authentication credentials and using authentication
clients to securely authenticate and send requests to external API providers.

## 1. Register NeMo Agent toolkit API Server as OAuth2.0 Client
To authenticate with a third-party API using OAuth 2.0, you must first register the application as a client with that
API provider. The NeMo Agent toolkit API server functions as both an API server and an OAuth 2.0
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
| **Grant Type(s)**     | The OAuth 2.0 flows the toolkit supports (for example, Authorization Code or Client Credential).         |
| **Scopes**            | The permissions your app is requesting (for example, `read:user` or `write:data`).       |

### Registering Redirect URIs for Development vs. Production Environments
**IMPORTANT**: Most OAuth providers require exact matches for redirect URIs.

| **Environment** | **Redirect URI Format**               |  **Notes**                         |
|-----------------|---------------------------------------|------------------------------------|
| Development     | `http://localhost:8000/auth/redirect` | Often used when testing locally.   |
| Production      | `https://yourdomain.com/auth/redirect`| Should use HTTPS and match exactly.|

### Configuring Registered App Credentials in Workflow Configuration YAML
After registering your application note the any credentials you need to use in the workflow configuration YAML file such as the client ID and client secret. These will be used in the next section when configuring the authentication provider.


## 2. Configuring Authentication Credentials
In the Workflow Configuration YAML file, user credentials required for API authentication are configured under the
`authentication` key. Users should provide all required and valid credentials for each authentication method to ensure
the library can authenticate requests without encountering credential related errors. Examples of currently supported
API configurations are
[OAuth 2.0 Authorization Code Grant Flow Configuration](../../../src/aiq/authentication/oauth2/oauth2_auth_code_flow_provider_config.py),
[API Key Configuration](../../../src/aiq/authentication/api_key/api_key_auth_provider_config.py), and [Basic HTTP Authentication](../../../src/aiq/authentication/http_basic_auth/register.py).

### Authentication YAML Configuration Example

The following example shows how to configure the authentication credentials for the OAuth 2.0 Authorization Code Grant Flow and API Key authentication. More information about each field can be queried using the `aiq info components -t auth_provider` command.

```yaml
authentication:
  test_auth_provider:
    _type: oauth2_auth_code_flow
    client_url: http://localhost:8000
    authorization_url: http://127.0.0.1:5000/oauth/authorize
    token_url: http://127.0.0.1:5000/oauth/token
    token_endpoint_auth_method: client_secret_post
    scopes:
      - openid
      - profile
      - email
    client_id: ${AIQ_OAUTH_CLIENT_ID}
    client_secret: ${AIQ_OAUTH_CLIENT_SECRET}
    use_pkce: false

  example_provider_name_api_key:
    _type: api_key
    raw_key: user_api_key
    header_name: accepted_api_header_name
    header_prefix: accepted_api_header_prefix
```

### OAuth2.0 Authorization Code Grant Configuration Reference
| Field Name                    | Description                                                                                                                        |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `example_provider_name_oauth` | A unique name used to identify the client credentials required to access the API provider.                                         |
| `_type`                       | Specifies the authentication type. For OAuth 2.0 Authorization Code Grant authentication, set this to `oauth2_auth_code_flow`. |
| `client_url`                  | URL of the OAuth 2.0 client server.                                                                                                |
| `authorization_url`           | URL used to initiate the authorization flow, where an authorization code is obtained to be later exchanged for an access token.    |
| `token_url`                   | URL used to exchange an authorization code for an access token and optional refresh token.                                         |
| `client_id`                   | The Identifier provided when registering the OAuth 2.0 client server with an API provider.                                         |
| `client_secret`               | A confidential string provided when registering the OAuth 2.0 client server with an API provider.                                  |
| `token_endpoint_auth_method`  | Some token provider endpoints require specific types of authentication. For example `client_secret_post`.                          |
| `scope`                       | List of permissions to the API provider (e.g., `read`, `write`).                                                                   |

### API Key Configuration Reference
| Field Name                      | Description                                                                                                |
|---------------------------------|------------------------------------------------------------------------------------------------------------|
| `example_provider_name_api_key` | A unique name used to identify the client credentials required to access the API provider.                 |
| `_type`                         | Specifies the authentication type. For API Key authentication, set this to `api_key`.                      |
| `raw_key`                       | API key value for authenticating requests to the API provider.                                             |
| `header_name`                   | The HTTP header used to transmit the API key for authenticating requests.                                  |
| `header_prefix`                 | Optional prefix for the HTTP header used to transmit the API key in authenticated requests (e.g., Bearer). |


## 3. Using the Authentication Provider
To use the authentication provider in your workflow, you can use the `AuthenticationRef` data model to retrieve the authentication provider from the `WorkflowBuilder` object.

### Sample Authentication Tool and Authentication Usage
```python
class WhoAmIConfig(FunctionBaseConfig, name="who_am_i"):
    """Find out who the currently logged in user is."""
    auth_provider: AuthenticationRef = Field(description="Reference to the authentication provider to use for authentication.")


@register_function(config_type=WhoAmIConfig)
async def auth_tool(config: WhoAmIConfig, builder: Builder):
    """
    Uses authentication to authenticate to any registered API provider.
    """
    auth_provider: AuthProviderBase = await builder.get_authentication(config.auth_provider)

    async def _arun(user_id: str) -> str:
        try:
            # Perform authentication (this will invoke the user authentication callback)
            auth_context: AuthResult = await auth_provider.authenticate(user_id=user_id)

            # With the auth context, we can make a request to the protected API resource
            async with httpx.AsyncClient(auth=auth_context) as client:
                response = await client.get("https://api.example.com/user")
                return response.json()

        except Exception as e:
            logger.exception("HTTP Basic authentication failed", exc_info=True)
            return f"HTTP Basic authentication for '{user_id}' failed: {str(e)}"

    yield FunctionInfo.from_fn(_arun, description="Find out who the currently logged in user is.")
```

## 4. Authentication by Application Configuration
Authentication methods not needing consent prompts, such as API Keys are supported uniformly across all deployment methods.
In contrast, support for methods that require user interaction can vary depending on the application's deployment and available
components. In some configurations, the system’s default browser handles the redirect directly, while in others, the
front-end UI is responsible for rendering the consent prompt in the browser.

Below is a table listing the current support for the various authentication methods based on the application

| # | Authentication Method                                | `aiq run` | `aiq serve` | Support Level                                         |
|---|------------------------------------------------------|-----------|-------------|-------------------------------------------------------|
| 1 | OAuth2.0 Authorization Code Grant Flow               | ✅         | ✅           | Full support with front-end UI only in websocket mode |
| 2 | API Key Authentication                               | ✅         | ✅           | Full support across all configurations                |
| 3 | HTTP Basic Authentication with Username and Password | ✅         | ❌           | Only available when using a console frontend          |

The sections below detail how OAuth2.0 authentication is handled in each supported configuration.

> ⚠️ **Important:**
> If using the OAuth2.0 Authorization Code Grant Flow, ensure that the `redirect_uri` in your workflow configuration matches the
> registered redirect URI in the API provider's console. Mismatched URIs will result in authentication failures. If you are using it
> in conjunction with the front-end UI, ensure that your browser supports popups and that the redirect URI is accessible from the browser.
