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
handled automatically on the user’s behalf, abstracting away the need for manual token management and authentication.
Currently supported authentication methods include OAuth 2.0 Authorization Code Flow and API keys. User authentication
credentials can be provided via the Workflow Configuration YAML file, allowing for centralized management and
configuration. The toolkit is only responsible for managing user credentials during workflow execution. Credentials are
loaded at runtime and accessed by the name specified in the YAML configuration file; they are not saved after workflow
termination.

## Configuring Authentication Credentials
In the Workflow Configuration YAML file, the authentication section stores the credentials required for API
authentication. Users should provide all required and valid credentials for each authentication method to ensure the
library can authenticate requests without encountering credential related errors. The authentication field example can
be found here [Authentication Configuration](#TODO EE ADD link to example file here.).

```yaml
authentication:
  example_provider_name_oauth:
    _type: oauth2
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

## OAuth2.0 Configuration Field Reference
| Field Name               | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `example_provider_name_oauth`    | A unique name used to identify the client credentials required to access the API provider.|
| `_type`                  | Specifies the authentication type. For OAuth 2.0 authentication, set this to `oauth2`.|
| `consent_prompt_mode`    | Specifies how the OAuth 2.0 client handles the consent prompt redirect.<br>•`browser` – Opens the system's default browser for user login.<br>•`frontend` – Redirects the consent prompt to the browser that initiated the POST request. A corresponding `consent_prompt_key` is required to receive the consent prompt redirect for the intended authentication provider.|
| `consent_prompt_key`       | A unique key used to retrieve the consent prompt redirect for the provider requesting authentication. Upon successful validation, the consent prompt redirect is returned to continue the OAuth 2.0 flow only if `frontend` is selected as the `consent_prompt_mode`.|
| `client_server_url`        | URL of the OAuth 2.0 client server. Please see [Client Server Guide](#TODO EE) on how to properly register api server as OAuth2.0 client server.|
| `authorization_url`        | URL used to initiate the authorization flow, where an authorization code is obtained to be later exchanged for an access token.|
| `authorization_token_url`  | URL used to exchange an authorization code for an access token and optional refresh token.|
| `client_id`                | The Identifier provided when registering the OAuth 2.0 client server with an API provider.|
| `client_secret`            | A confidential string provided when registering the OAuth 2.0 client server with an API provider.|
| `audience`                 | Represents the specific API provider for which the authorization tokens were issued (provider-specific).|
| `scope`                    | List of permissions to the API provider (e.g., `read`, `write`).|

## API Key Configuration Field Reference
| Field Name               | Description                                                 |
|--------------------------|-------------------------------------------------------------|
| `example_provider_name_api_key`| A unique name used to identify the client credentials required to access the API provider.|
| `_type`                  | Specifies the authentication type. For API Key authentication, set this to `api_key`.|
| `api_key`                | API key value for authenticating requests to the API provider.|
| `header_name`            | The HTTP header used to transmit the API key for authenticating requests.|
| `header_prefix`          | Optional prefix for the HTTP header used to transmit the API key in authenticated requests (e.g., Bearer).|

## Registering Agent Intelligence Toolkit API Server as OAuth2.0 Client
The Agent Intelligence Toolkit API server doubles as an OAuth 2.0 client, meaning it can be registered with an
API provider to initiate the OAuth flow, manage token exchanges, and serve requests through its exposed endpoints. This
section outlines a general approach for registering the API server as an OAuth 2.0 client with your API provider in
order to enable authentication using OAuth 2.0.

### 1. Access the API Provider’s Developer Console
Navigate to the API provider’s developer or cloud console and follow the instructions to register the API
server as an authorized application.

## Configuring OAuth2.0 Authentication Credential for Application Environment

## Using API Request Utility Function
