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

# Adding an API Authentication Provider to NeMo Agent Toolkit
:::{note}
We recommend reading the [Streamlining API Authentication](../reference/api-authentication.md) guide before proceeding with
this detailed documentation.
:::

The NeMo Agent toolkit offers a set of built-in authentication providers for accessing API resources. Additionally, it includes
a plugin system that allows developers to define and integrate custom authentication providers.

## Existing API Authentication Providers
You can view the list of existing API Authentication Providers by running the following command:
```bash
aiq info components -t authentication_provider
```

## Provider Types
In the NeMo Agent toolkit, the providers (credentials) required to authenticate with an API resource are defined separately
from the clients that facilitate the authentication process. Authentication providers, such as `APIKeyConfig` and
`AuthCodeGrantConfig`, store the authentication credentials, while clients like `APIKeyClient` and
`AuthCodeGrantClient` use those credentials to perform authentication.

## Extending an API Authentication Provider
The first step in adding an authentication provider is to create a configuration model that inherits from
{class}`aiq.data_models.authentication.AuthenticationBaseConfig` class and define the credentials required to
authenticate with the target API resource.

The following example shows how to define and register a custom evaluator and can be found here:
{class}`aiq.authentication.oauth2.OAuth2AuthorizationCodeFlowConfig` class:
```python
class OAuth2AuthorizationCodeFlowConfig(AuthenticationBaseConfig, name="oauth2_auth_code_flow"):

    model_config = ConfigDict(extra="forbid")

    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The secret associated with the client_id.")
    authorization_url: str = Field(description="The authorization URL for OAuth 2.0 authentication.")
    token_url: str = Field(description="The token URL for OAuth 2.0 authentication.")
    token_endpoint_auth_method: str | None = Field(description="The authentication method for the token endpoint.",
                                                   default=None)
    scopes: list[str] = Field(description="The space-delimited scopes for OAuth 2.0 authentication.",
                              default_factory=list)

    # Configuration for the local server that handles the redirect
    client_url: str = Field(description="The base URL for the client application.", default="http://localhost:8000")
    run_local_redirect_server: bool = Field(default=False,
                                            description="Whether to run a local server to handle the redirect URI.")
    local_redirect_server_port: int = Field(default=8000,
                                            description="Port for the local redirect "
                                            "server to listen on.")
    redirect_path: str = Field(default="/auth/redirect",
                               description="Path for the local redirect server to handle the callback.")
    use_pkce: bool = Field(default=False,
                           description="Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow.")

    @property
    def redirect_uri(self) -> str:
        return f"{self.client_url}{self.redirect_path}"
```

### Registering the Provider
An asynchronous function decorated with {py:deco}`aiq.cli.register_workflow.register_authentication_provider` is used to
register the provider with NeMo Agent toolkit by yielding an instance of
{class}`aiq.builder.authentication.AuthenticationProviderInfo`.

The `OAuth2AuthorizationCodeFlowConfig` from the previous section is registered as follows:
```python
@register_authentication_provider(config_type=OAuth2AuthorizationCodeFlowConfig)
async def oauth2(authentication_provider: OAuth2AuthorizationCodeFlowConfig, builder: Builder):
    yield AuthenticationProviderInfo(config=authentication_provider, description="OAuth 2.0 authentication provider.")
```
## Extending the API Authentication Client
As described earlier, each API authentication provider defines the credentials and parameters required to authenticate
with a specific API service. A corresponding API authentication client uses this configuration to initiate and
complete the authentication process. NeMo Agent toolkit provides an extensible base class `AuthenticationClientBase`
to simplify the development of custom authentication clients for various authentication methods.
 These base classes provide a structured interface for implementing key functionality, including:
- Validating configuration credentials
- Interfacing with the NeMo Agent toolkit frontend authentication flow handlers
- Returning appropriate authentication tokens or credentials

To implement a custom client, extend the appropriate base class and override the required methods. For detailed
documentation on the methods and expected behavior, refer to the docstrings provided in the
{py:deco}`aiq.authentication.interfaces` module.

## Registering the API Authentication Client
To register an authentication client, define an asynchronous function decorated
with {py:deco}`aiq.cli.register_workflow.register_authentication_client`. The `register_authentication_client`
decorator requires a single argument: `config_type`, which specifies the authentication configuration class associated
with the provider.

`src/aiq/authentication/oauth2/register.py`:
```python
@register_authentication_client(config_type=OAuth2AuthorizationCodeFlowConfig)
async def oauth2_client(authentication_provider: OAuth2AuthorizationCodeFlowConfig, builder: Builder):
    yield OAuth2Client(authentication_provider)
```
Similar to the registration function for the provider, the client registration function can perform any necessary setup
actions before yielding the client, along with cleanup actions after the `yield` statement.

## Testing an Authentication Client
After implementing a new authentication client, it’s important to verify that the required functionality works as
expected. This can be done by writing integration tests. It is important to minimize the amount of mocking in the tests
to ensure that the client behaves as expected in a real-world scenario. You can find examples of existing tests in the repository
at `tests/aiq/authentication`.

## Packaging the Provider and Client

The provider and client will need to be bundled into a Python package, which in turn will be registered with AIQ
toolkit as a [plugin](../extend/plugins.md). In the `pyproject.toml` file of the package the
`project.entry-points.'aiq.components'` section, defines a Python module as the entry point of the plugin. Details on
how this is defined are found in the [Entry Point](../extend/plugins.md#entry-point) section of the plugins document.
By convention, the entry point module is named `register.py`, but this is not a requirement.

In the entry point module, it is important that the provider is defined first followed by the client. This ensures that
the provider is added to the NeMo Agent toolkit registry before the client is registered. A hypothetical `register.py` file
could be defined as follows:

```python
# We need to ensure that the provider is registered prior to the client

import register_provider
import register_client
```
