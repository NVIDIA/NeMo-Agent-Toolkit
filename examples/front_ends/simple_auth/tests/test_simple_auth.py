# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
import typing

import pytest

if typing.TYPE_CHECKING:
    import requests.cookies

    from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
    from nat.data_models.authentication import AuthenticatedContext
    from nat.data_models.authentication import AuthFlowType


async def _auth_callback(cookies: "requests.cookies.RequestsCookieJar",
                         config: "OAuth2AuthCodeFlowProviderConfig",
                         method: "AuthFlowType") -> "AuthenticatedContext":
    """
    Simulate the auth callback that would normally be added by the CLI front-end defined in:
    src/nat/front_ends/console/authentication_flow_handler.py

    That handler launches a web browser for the user to authenticate interactively, which then redirects to a local
    webserver, we will simulate here without doing either.
    """
    import secrets

    import requests
    from authlib.integrations.httpx_client import AsyncOAuth2Client

    from nat.authentication.oauth2.oauth2_auth_code_flow_provider_config import OAuth2AuthCodeFlowProviderConfig
    from nat.data_models.authentication import AuthenticatedContext
    from nat.data_models.authentication import AuthFlowType

    assert isinstance(config, OAuth2AuthCodeFlowProviderConfig)
    assert method == AuthFlowType.OAUTH2_AUTHORIZATION_CODE

    state = secrets.token_urlsafe(16)
    oauth_client = AsyncOAuth2Client(client_id=config.client_id,
                                     client_secret=config.client_secret.get_secret_value(),
                                     redirect_uri=config.redirect_uri,
                                     scope=" ".join(config.scopes),
                                     token_endpoint=config.token_url,
                                     token_endpoint_auth_method=config.token_endpoint_auth_method)
    auth_url, ___ = oauth_client.create_authorization_url(url=config.authorization_url, state=state)

    response = requests.post(auth_url,
                             params={
                                 "response_type": "code",
                                 "client_id": config.client_id,
                                 "scope": ["openid", "profile", "email"],
                                 "state": state
                             },
                             cookies=cookies,
                             headers={"Content-Type": "application/x-www-form-urlencoded"},
                             data=[("confirm", "on")],
                             allow_redirects=False,
                             timeout=30)
    response.raise_for_status()
    redirect_location = response.headers["Location"]

    token = await oauth_client.fetch_token(  # type: ignore[arg-type]
        url=config.token_url,
        authorization_response=redirect_location,
        code_verifier=None,
        state=state,
    )

    return AuthenticatedContext(
        headers={"Authorization": f"Bearer {token['access_token']}"},
        metadata={
            "expires_at": token.get("expires_at"), "raw_token": token
        },
    )


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "restore_environ")
async def test_full_workflow(oauth2_client_credentials: dict[str, str]):
    import urllib

    from pydantic import SecretStr

    from nat.runtime.loader import load_config
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_simple_auth.ip_lookup import WhoAmIConfig

    # Even though we set this later on the config object, the yaml won't validate without these env vars set
    os.environ.update({
        "NAT_OAUTH_CLIENT_ID": oauth2_client_credentials["id"],
        "NAT_OAUTH_CLIENT_SECRET": oauth2_client_credentials["secret"],
    })

    config_file = locate_example_config(WhoAmIConfig)
    config = load_config(config_file)

    # Override the oauth URLs to point to the test OAuth2 server
    oauth_url = oauth2_client_credentials["url"]
    allowed_origins = config.general.front_end.cors.allow_origins
    for (i, url) in enumerate(allowed_origins):
        if urllib.parse.urlparse(url).port == 5001:
            allowed_origins[i] = oauth_url

    # Override the configs with the test OAuth2 server details
    config.authentication['test_auth_provider'].authorization_url = f"{oauth_url}/oauth/authorize"
    config.authentication['test_auth_provider'].token_url = f"{oauth_url}/oauth/token"
    config.authentication['test_auth_provider'].client_id = oauth2_client_credentials["id"]
    config.authentication['test_auth_provider'].client_secret = SecretStr(oauth2_client_credentials["secret"])
    config.functions['who_am_i_function'].api_url = f"{oauth_url}/api/me"

    await run_workflow(config=config,
                       question="Who am I logged in as?",
                       session_kwargs={
                           "user_authentication_callback":
                               functools.partial(_auth_callback, oauth2_client_credentials["cookies"])
                       },
                       expected_answer=oauth2_client_credentials["username"])
