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
We recommend reading the [Streamlining API Authentication](../reference/api-authentication.md) guide before proceeding with this detailed documentation.
:::

The NeMo Agent toolkit offers a set of built-in authentication providers for accessing API resources. Additionally, it includes
a plugin system that allows developers to define and integrate custom authentication providers.

## Existing API Authentication Providers
You can view the list of existing API Authentication Providers by running the following command:
```bash
aiq info components -t auth_provider
```

## Provider Types
In the NeMo Agent toolkit, the providers (credentials) required to authenticate with an API resource are defined separately
from the clients that facilitate the authentication process. Authentication providers, such as `APIKeyAuthProviderConfig` and
`OAuth2AuthCodeFlowProviderConfig`, store the authentication credentials, while clients like `APIKeyAuthProvider` and
`OAuth2AuthCodeFlowProvider` use those credentials to perform authentication.

## Extending an API Authentication Provider
The first step in adding an authentication provider is to create a configuration model that inherits from the
{py:class}`~aiq.data_models.authentication.AuthProviderBaseConfig` class and define the credentials required to
authenticate with the target API resource.

The following example shows how to define and register a custom evaluator and can be found here:
{py:class}`~aiq.authentication.oauth2.oauth2_auth_code_flow_provider_config.OAuth2AuthCodeFlowProviderConfig` class:
```python
class OAuth2AuthCodeFlowProviderConfig(AuthProviderBaseConfig, name="oauth2_auth_code_flow"):

    client_id: str = Field(description="The client ID for OAuth 2.0 authentication.")
    client_secret: str = Field(description="The secret associated with the client_id.")
    authorization_url: str = Field(description="The authorization URL for OAuth 2.0 authentication.")
    token_url: str = Field(description="The token URL for OAuth 2.0 authentication.")
    token_endpoint_auth_method: str | None = Field(
        description=("The authentication method for the token endpoint. "
                     "Usually one of `client_secret_post` or `client_secret_basic`."),
        default=None)
    redirect_uri: str = Field(description="The redirect URI for OAuth 2.0 authentication. Must match the registered "
                              "redirect URI with the OAuth provider.")
    scopes: list[str] = Field(description="The scopes for OAuth 2.0 authentication.", default_factory=list)
    use_pkce: bool = Field(default=False,
                           description="Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow.")

    authorization_kwargs: dict[str, str] | None = Field(description=("Additional keyword arguments for the "
                                                                     "authorization request."),
                                                        default=None)
```

### Registering the Provider
An asynchronous function decorated with {py:func}`~aiq.cli.register_workflow.register_auth_provider` is used to register the provider with NeMo Agent toolkit by yielding an instance of
{py:class}`~aiq.authentication.interfaces.AuthProviderBase`.

The `OAuth2AuthCodeFlowProviderConfig` from the previous section is registered as follows:
```python
@register_auth_provider(config_type=OAuth2AuthCodeFlowProviderConfig)
async def oauth2_client(authentication_provider: OAuth2AuthCodeFlowProviderConfig, builder: Builder):
    from aiq.authentication.oauth2.oauth2_auth_code_flow_provider import OAuth2AuthCodeFlowProvider

    yield OAuth2AuthCodeFlowProvider(authentication_provider)
```

## Defining the Provider
Each authentication provider should inherit from the {py:class}`~aiq.authentication.interfaces.AuthProviderBase` class, and implement the required methods.

## Testing the new Provider
After implementing a new authentication provider, it’s important to verify that the required functionality works as expected. This can be done by writing integration tests. It is important to minimize the amount of mocking in the tests to ensure that the provider behaves as expected in a real-world scenario. You can find examples of existing tests in the repository at `tests/aiq/authentication`.

## Packaging the Provider

The provider will need to be bundled into a Python package, which in turn will be registered with the toolkit as a [plugin](../extend/plugins.md). In the `pyproject.toml` file of the package the
`project.entry-points.'aiq.components'` section, defines a Python module as the entry point of the plugin. Details on how this is defined are found in the [Entry Point](../extend/plugins.md#entry-point) section of the plugins document. By convention, the entry point module is named `register.py`, but this is not a requirement.

In the entry point module, the registration of provider, that is the function decorated with `register_auth_provider`, needs to be defined, either directly or imported from another module. A hypothetical `register.py` file could be defined as follows:

```python
import register_provider
```
