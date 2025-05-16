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

from aiq.authentication.exceptions import OAuthCodeFlowError
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ResponseManager:

    def __init__(self) -> None:
        pass

    async def _handle_oauth_authorization_response_codes(
            self, response: httpx.Response, authentication_provider: OAuth2Config) -> None:  # TODO EE: Update.
        """
        Handles various OAuth2.0 authorization responses.

        Args:
            response (httpx.Response): The HTTP response from the authentication server.
        """
        try:

            if response.status_code == 302:
                redirect_location_header: str | None = response.headers.get("Location")

                if not redirect_location_header:
                    raise OAuthCodeFlowError(
                        "Missing 'Location' header in 302 response to redirect user to consent browser.")

                await self._handle_oauth_consent_browser(redirect_location_header, authentication_provider)

            if response.status_code in (400, 401, 403):
                await self._oauth_400_status_code_handler(response)

        except Exception as e:
            logger.error("Unexpected error occured while handling authorization request response: %s",
                         str(e),
                         exc_info=True)
            raise OAuthCodeFlowError("Unexpected error occured while handling authorization request response") from e

    async def _handle_oauth_consent_browser(self, location_header: str, authentication_provider: OAuth2Config) -> None:
        """
        Handles the consent prompt redirect for different run environments.

        Args: # TODO EE: Update
            location_header (str) : Location header from authorization server HTTP 302 consent prompt redirect.
            authentication_provider (OAuth2Config): The registered OAuth2.0 authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        try:
            if authentication_provider.consent_prompt_mode == ConsentPromptMode.BROWSER:

                default_browser = webbrowser.get()
                default_browser.open(location_header)

            if authentication_provider.consent_prompt_mode == ConsentPromptMode.FRONTEND:

                logger.info("A POST request is required to retrieve the 302 redirect. "
                            "Please ensure your client sends a POST request to continue OAuth2.0 code flow. ")

                authentication_provider.consent_prompt_location_url = location_header

                await _CredentialsManager()._wait_for_consent_prompt_url()

                authentication_provider.consent_prompt_location_url = None

        except webbrowser.Error as e:
            logger.error("Unable to open defualt browser: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Unable to complete OAuth2.0 process.") from e

        except Exception as e:
            logger.error("Exception occured: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Unable to complete OAuth2.0 process.") from e

    async def _oauth_400_status_code_handler(self, response: httpx.Response):
        """
        Handles the response to a protected resource request with an authentication attempt using
        an expired or invalid access token. According to RFC 6750 When a request fails, the resource server
        responds using the appropriate HTTP status code (typically, 400, 401, 403, or 405) and
        includes one of the following error codes in the response: invalid_request, invalid_token,
        insufficient_scope.

        Args:
            response (httpx.Response): The response form the OAuth2.0 authentication server.
        """
        # 400 Bad Request: Invalid refresh token provided or malformed request.
        if response.status_code == 400:
            pass

        # 401 Unauthorized: Token is missing, revoked, invlaid or expired.
        if response.status_code == 401:
            pass

        if response.status_code == 403:  # TODO EE: Update and add tests.
            pass
