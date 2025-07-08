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

# Adding an API Authentication Provider to NVIDIA Agent Intelligence Toolkit
:::{note}
We recommend reading the [Streamlining API Authentication](../reference/api-authentication.md) guide before proceeding with
this detailed documentation.
:::
The AIQ Toolkit offers a set of built-in authentication providers for accessing API resources. Additionally, it includes
a plugin system that allows developers to define and integrate custom authentication providers.

## Existing API Authentication Providers
You can view the list of existing API Authentication Providers by running the following command:
```bash
aiq info components -t authentication_provider
```

## Provider Types
In the AIQ Toolkit, the providers (credentials) required to authenticate with an API resource are defined separately
from the clients that facilitate the authentication process. Authentication providers, such as `APIKeyConfig` and
`AuthCodeGrantConfig`, store the authentication credentials, while clients like `APIKeyClient` and
`AuthCodeGrantClient` use those credentials to perform authentication.

## Extending an API Authentication Provider
The first step in adding an authentication provider is to create a configuration model that inherits from
{class}`aiq.data_models.authentication.AuthenticationBaseConfig` class and define the credentials required to
authenticate with the target API resource.

The following example shows how to define and register a custom evaluator and can be found here:
{class}`aiq.authentication.oauth2.AuthCodeGrantConfig` class:
```python
class AuthCodeGrantConfig(OAuthUserConsentConfigBase, name="oauth2_authorization_code_grant"):
    """
    OAuth 2.0 authorization code grant authentication configuration model.
    """
    client_server_url: str = Field(description="The base url of the API server instance. "
                                   "This is needed to properly construct the redirect uri i.e: http://localhost:8000")
    authorization_url: str = Field(description="The base url to the authorization server in which authorization "
                                   "requests are made to receive access codes.")
    authorization_token_url: str = Field(
        description="The base url to the authorization token server in which access codes "
        "are exchanged for access tokens.")
```

### Registering the Provider
An asynchronous function decorated with {py:deco}`aiq.cli.register_workflow.register_authentication_provider` is used to
register the provider with AIQ toolkit by yielding an instance of
{class}`aiq.builder.authentication.AuthenticationProviderInfo`.

The `AuthCodeGrantConfig` from the previous section is registered as follows:
`src/aiq/authentication/oauth2/auth_code_grant_config.py`:
```python
@register_authentication_provider(config_type=AuthCodeGrantConfig)
async def oauth2_authorization_code_grant(authentication_provider: AuthCodeGrantConfig, builder: Builder):

    yield AuthenticationProviderInfo(config=authentication_provider,
                                     description="OAuth 2.0 Authorization Code Grant authentication provider.")
```
## Extending the API Authentication Client
As described earlier, each API authentication provider defines the credentials and parameters required to authenticate
with a specific API service. A corresponding API authentication client uses this configuration to initiate and
complete the authentication process. AIQ Toolkit provides two extensible base classes `AuthenticationClientBase` and
 `OAuthClientBase` to simplify the development of custom authentication clients for various authentication methods.
 These base classes provide a structured interface for implementing key functionality, including:
- Validating configuration credentials
- Constructing authenticated request parameters
- Completing OAuth flows across different execution environments

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
@register_authentication_client(config_type=AuthCodeGrantConfig)
async def oauth2_authorization_code_grant_client(authentication_provider: AuthCodeGrantConfig, builder: Builder):

    yield AuthCodeGrantClient(config=authentication_provider)
```
Similar to the registration function for the provider, the client registration function can perform any necessary setup
actions before yielding the client, along with cleanup actions after the `yield` statement.

## Testing an Authentication Client
After implementing a new authentication client, it’s important to verify that the required functionality works as
expected. This can be done by writing integration tests. To test a standard authentication client, use the
`AuthenticationClientTester` class located in the {py:mod}`tests.aiq.authentication.test_custom_authentication_client`
module. For clients that implement OAuth flows, use the `OAuth2FlowTester` class provided in the
{py:mod}`tests.aiq.authentication.oauth2_mock_server` module.

```python
async def test_api_key_client_integration():
    """Authentication Client Integration tests."""
    from test_custom_authentication_client import AuthenticationClientTester

    client = APIKeyClient(config=APIKeyConfig(raw_key="test_api_key_12345",
                                              header_name="X-API-Key",
                                              header_prefix="Bearer"),
                          config_name="test_api_key")

    tester = AuthenticationClientTester(auth_client=client)

    # Run the complete Authentication Client integration test suite
    assert await tester.run() is True
```

```python
async def test_oauth2_full_flow():
    """Test the complete OAuth2 authorization code flow."""
    # Create OAuth2FlowTester instance with a minimal config and client for testing
    minimal_config = AuthCodeGrantConfig(client_server_url="https://test.com",
                                         authorization_url="https://test.com/auth",
                                         authorization_token_url="https://test.com/token",
                                         consent_prompt_key="test_key_secure",
                                         client_secret="test_secret_secure_16_chars_minimum",
                                         client_id="test_client",
                                         audience="test_audience",
                                         scope=["test_scope"])

    auth_client = AuthCodeGrantClient(config=minimal_config, config_name="test_config")

    tester = OAuth2FlowTester(oauth_client=auth_client, flow=OAuth2Flow.AUTHORIZATION_CODE)

    # Run the complete OAuth2 flow test suite
    assert await tester.run() is True
```

## Packaging the Provider and Client

The provider and client will need to be bundled into a Python package, which in turn will be registered with AIQ
toolkit as a [plugin](../extend/plugins.md). In the `pyproject.toml` file of the package the
`project.entry-points.'aiq.components'` section, defines a Python module as the entry point of the plugin. Details on
how this is defined are found in the [Entry Point](../extend/plugins.md#entry-point) section of the plugins document.
By convention, the entry point module is named `register.py`, but this is not a requirement.

In the entry point module it is important that the provider is defined first followed by the client, this ensures that
the provider is added to the AIQ toolkit registry before the client is registered. A hypothetical `register.py` file
could be defined as follows:

```python
# We need to ensure that the provider is registered prior to the client

import register_provider
import register_client
```
