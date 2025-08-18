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

:::{warning}
**Experimental Feature**: The Authentication Provider API is experimental and may change in future releases. Future versions may introduce breaking changes without notice.
:::

The NeMo Agent toolkit simplifies API authentication by streamlining credential management and validation, enabling secure
access to API providers across a variety of runtime environments. This functionality allows users to authenticate with
protected API resources directly from workflow tools, abstracting away low-level authentication logic and enabling
greater focus on data retrieval and processing. Users can define multiple authentication providers in their workflow
configuration file, each uniquely identified by a provider name. Authentication is supported in headless and server modes. Credentials are
securely loaded into memory at runtime, accessed by provider name, and are never logged or persisted. They are available only during workflow execution to ensure secure and centralized handling. Currently supported authentication configurations include OAuth 2.0 Authorization Code Grant Flow and API keys, each managed by dedicated authentication clients. The system is designed
for extensibility, allowing developers to introduce new credential types and clients to support additional
authentication methods and protected API access patterns.

## API Authentication Configuration and Usage Walkthrough
This guide provides a step-by-step walkthrough for configuring authentication credentials and using authentication
clients to securely authenticate and send requests to external API providers.

## 1. Register NeMo Agent Toolkit API Server as an OAuth2.0 Client
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
| **Redirect URIs**   | The URIs where the API will redirect users after authorization.               |
| **Grant Types**     | The OAuth 2.0 flows the toolkit supports (for example, Authorization Code or Client Credential).         |
| **Scopes**            | The permissions your app is requesting (for example, `read:user` or `write:data`).       |

### Registering Redirect URIs for Development vs. Production Environments
**IMPORTANT**: Most OAuth providers require exact matches for redirect URIs.

| **Environment** | **Redirect URI Format**               |  **Notes**                         |
|-----------------|---------------------------------------|------------------------------------|
| Development     | `http://localhost:8000/auth/redirect` | Often used when testing locally.   |
| Production      | `https://<yourdomain>/auth/redirect`  | Should use HTTPS and match exactly.|

### Configuring Registered App Credentials in Workflow Configuration YAML
After registering your application note the any credentials you need to use in the workflow configuration YAML file such as the client ID and client secret. These will be used in the next section when configuring the authentication provider.


## 2. Configuring Authentication Credentials
In the workflow configuration YAML file, user credentials required for API authentication are configured under the
`authentication` key. Users should provide all required and valid credentials for each authentication method to ensure
the library can authenticate requests without encountering credential related errors. Examples of currently supported
API configurations are
[OAuth 2.0 Authorization Code Grant Flow Configuration](../../../src/nat/authentication/oauth2/oauth2_auth_code_flow_provider_config.py),
[API Key Configuration](../../../src/nat/authentication/api_key/api_key_auth_provider_config.py), and [Basic HTTP Authentication](../../../src/nat/authentication/http_basic_auth/register.py).

### Authentication YAML Configuration Example

The following example shows how to configure the authentication credentials for the OAuth 2.0 Authorization Code Grant Flow and API Key authentication. More information about each field can be queried using the `nat info components -t auth_provider` command.

```yaml
authentication:
  test_auth_provider:
    _type: oauth2_auth_code_flow
    authorization_url: http://127.0.0.1:5000/oauth/authorize
    token_url: http://127.0.0.1:5000/oauth/token
    token_endpoint_auth_method: client_secret_post
    scopes:
      - openid
      - profile
      - email
    client_id: ${NAT_OAUTH_CLIENT_ID}
    client_secret: ${NAT_OAUTH_CLIENT_SECRET}
    use_pkce: false

  example_provider_name_api_key:
    _type: api_key
    raw_key: user_api_key
    custom_header_name: accepted_api_header_name
    custom_header_prefix: accepted_api_header_prefix
```

### OAuth2.0 Authorization Code Grant Configuration Reference
| Field Name | Description |
|-------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `test_auth_provider` | A unique name used to identify the client credentials required to access the API provider. |
| `_type` | Specifies the authentication type. For OAuth 2.0 Authorization Code Grant authentication, set this to `oauth2_auth_code_flow`. |
| `client_id` | The Identifier provided when registering the OAuth 2.0 client server with an API provider. |
| `client_secret` | A confidential string provided when registering the OAuth 2.0 client server with an API provider. |
| `authorization_url` | URL used to initiate the authorization flow, where an authorization code is obtained to be later exchanged for an access token. |
| `token_url` | URL used to exchange an authorization code for an access token and optional refresh token. |
| `token_endpoint_auth_method` | Some token provider endpoints require specific types of authentication. For example `client_secret_post`. |
| `redirect_uri` | The redirect URI for OAuth 2.0 authentication. Must match the registered redirect URI with the OAuth provider.|
| `scopes` | List of permissions to the API provider (e.g., `read`, `write`). |
| `use_pkce` | Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow, defaults to `False` |
| `authorization_kwargs` | Additional keyword arguments to include in the authorization request. |


### API Key Configuration Reference
| Field Name | Description |
|---------------------------------|------------------------------------------------------------------------------------------------------------|
| `example_provider_name_api_key` | A unique name used to identify the client credentials required to access the API provider. |
| `_type` | Specifies the authentication type. For API Key authentication, set this to `api_key`. |
| `raw_key` | API key value for authenticating requests to the API provider. |
| `auth_scheme` | The HTTP authentication scheme to use. Supported schemes: `BEARER`, `X_API_KEY`, `BASIC`, and `CUSTOM`, default is `BEARER` |
| `custom_header_name` | The HTTP header used to transmit the API key for authenticating requests. |
| `custom_header_prefix` | Optional prefix for the HTTP header used to transmit the API key in authenticated requests (e.g., Bearer). |


## 3. Using the Authentication Provider
To use the authentication provider in your workflow, you can use the `AuthenticationRef` data model to retrieve the authentication provider from the `WorkflowBuilder` object.

### Sample Authentication Tool and Authentication Usage
```python
class WhoAmIConfig(FunctionBaseConfig, name="who_am_i"):
    """
    Function that looks up the user's identity.
    """
    auth_provider: AuthenticationRef = Field(description=("Reference to the authentication provider to use for "
                                                          "authentication before making the who am i request."))

    api_url: str = Field(default="http://localhost:5001/api/me", description="Base URL for the who am i API")
    timeout: int = Field(default=10, description="Request timeout in seconds")
```

Full source code for the above example can be found in `examples/front_ends/simple_auth/src/nat_simple_auth/ip_lookup.py`.

## 4. Authentication by Application Configuration
Authentication methods not needing consent prompts, such as API Keys are supported uniformly across all deployment methods.
In contrast, support for methods that require user interaction can vary depending on the application's deployment and available
components. In some configurations, the system’s default browser handles the redirect directly, while in others, the
front-end UI is responsible for rendering the consent prompt.

Below is a table listing the current support for the various authentication methods based on the application

| # | Authentication Method                                | `nat run` | `nat serve` | Support Level                                         |
|---|------------------------------------------------------|-----------|-------------|-------------------------------------------------------|
| 1 | OAuth2.0 Authorization Code Grant Flow               | ✅         | ✅           | Full support with front-end UI only in websocket mode |
| 2 | API Key Authentication                               | ✅         | ✅           | Full support across all configurations                |
| 3 | HTTP Basic Authentication with Username and Password | ✅         | ❌           | Only available when using a console frontend          |

The sections below detail how OAuth2.0 authentication is handled in each supported configuration.

> ⚠️ **Important:**
> If using the OAuth2.0 Authorization Code Grant Flow, ensure that the `redirect_uri` in your workflow configuration matches the
> registered redirect URI in the API provider's console. Mismatched URIs will result in authentication failures. If you are using it
> in conjunction with the front-end UI, ensure that your browser supports popups and that the redirect URI is accessible from the browser.
