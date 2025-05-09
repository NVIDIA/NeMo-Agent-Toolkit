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

from aiq.authentication.exceptions import OAuthError
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ResponseManager:

    def __init__(self) -> None:
        pass

    async def _handle_oauth_authorization_response(self,
                                                   response: httpx.Response | None,
                                                   authentication_provider: OAuth2Config) -> None:
        """
        Handles an OAuth authorization response to extract and handle the redirect URL.

        Args:
            response (httpx.Response): The HTTP response from the authorization request.
        """
        try:
            if response is None:
                raise OAuthError("Invalid response from authorization request.")

            if response.status_code == 302:
                redirect_location_header: str | None = response.headers.get("Location")

                if not redirect_location_header:
                    raise OAuthError("Missing 'Location' header in 302 response to redirect user to consent browser.")

                await self._handle_oauth_consent_browser(redirect_location_header, authentication_provider)

        except Exception as e:
            logger.error("Unexpected error occured while handling authorization request response: %s",
                         str(e),
                         exc_info=True)
            raise OAuthError("Unexpected error occured while handling authorization request response") from e

    async def _handle_oauth_consent_browser(self, location_header: str, authentication_provider: OAuth2Config) -> None:
        """
        Handles the consent prompt redirect for different run environments.

        Args:
            location_header (str) : Location header from authorization server HTTP 302 consent prompt redirect.
            authentication_provider (OAuth2Config): The registered OAuth2.0 authentication provider.
        """
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
