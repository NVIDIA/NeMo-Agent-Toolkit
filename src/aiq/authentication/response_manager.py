# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
import webbrowser

import httpx

from aiq.authentication.oauth2_authenticator import OAuthError
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ResponseManager:

    def __init__(self) -> None:
        pass

    async def _handle_oauth_authorization_response(self,
                                                   response: httpx.Response | None,
                                                   authentication_provider: str | None) -> None:
        """
        Handles an OAuth authorization response to extract and handle the redirect URL.

        Args:
            response (httpx.Response): The HTTP response from the authorization request.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        try:
            auth_provider: OAuth2Config | None = _CredentialsManager()._get_authentication_provider(
                authentication_provider)

            if auth_provider is None:
                raise ValueError(f"Authorization provider not found: {authentication_provider}")

            if not isinstance(auth_provider, OAuth2Config):
                raise TypeError(f"Authorization provider: {authentication_provider} not of type: {OAuth2Config}.")

            # TODO EE: Update a Status handler function
            if response.status_code == 302:
                redirect_location_header: str | None = response.headers.get("Location")

                if not redirect_location_header:
                    raise OAuthError("Missing 'Location' header in 302 response to redirect user to consent browser.")

                await self._handle_oauth_consent_browser(redirect_location_header, auth_provider)

        except (ValueError, Exception) as e:  # TODO EE: Update Error conditions.
            logger.error("Unable to open defualt browser: %s", str(e), exc_info=True)

    async def _handle_oauth_consent_browser(
            self, location_header: str, authentication_provider: OAuth2Config) -> None:  # TODO EE: Update doc string
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            if authentication_provider.consent_prompt_mode == ConsentPromptMode.BROWSER:
                default_browser = webbrowser.get()
                default_browser.open(location_header)

            if authentication_provider.consent_prompt_mode == ConsentPromptMode.POLLING:
                await _CredentialsManager()._wait_for_consent_prompt_url()

        except webbrowser.Error as e:
            logger.error("Unable to open defualt browser: %s", str(e), exc_info=True)
            raise OAuthError("Unable to complete OAuth2.0 process.") from e

        except Exception as e:
            logger.error("Exception occured: %s", str(e), exc_info=True)
            raise OAuthError("Unable to complete OAuth2.0 process.") from e
