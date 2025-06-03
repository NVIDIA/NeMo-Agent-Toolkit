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
from aiq.front_ends.fastapi.message_handler import MessageHandler

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ResponseManager:

    def __init__(self) -> None:
        self._message_handler: MessageHandler | None = None

    @property
    def message_handler(self) -> MessageHandler | None:
        """
        Get the message handler instance.

        Returns:
            MessageHandler | None: Returns the message handler instance if set, otherwise None.
        """
        return self._message_handler

    @message_handler.setter
    def message_handler(self, message_handler: MessageHandler) -> None:
        """
        Set the message handler member variable.

        Args:
            message_handler (MessageHandler): The message handler to be set.
        """
        self._message_handler = message_handler

    async def _handle_oauth_authorization_response_codes(self,
                                                         response: httpx.Response,
                                                         authentication_provider: OAuth2Config) -> None:
        """
        Handles various OAuth2.0 authorization responses.

        Args:
            response (httpx.Response): The HTTP response from the authentication server.
            authentication_provider (OAuth2Config): The registered OAuth2.0 authentication provider.
        """
        try:
            # Handles the 302 redirect status code from OAuth2.0 authorization server.
            if response.status_code == 302:
                redirect_location_header: str | None = response.headers.get("Location")

                if not redirect_location_header:
                    raise OAuthCodeFlowError(
                        "Missing 'Location' header in 302 response to redirect user to consent browser.")

                await self._handle_oauth_302_consent_browser(redirect_location_header, authentication_provider)

            # Handles the 4xx client status codes from OAuth2.0 authorization server.
            elif response.status_code >= 400 and response.status_code < 500:
                await self._oauth_400_status_code_handler(response)

            else:
                raise OAuthCodeFlowError(f"Unknown response code: {response.status_code}. Response: {response.text}")

        except Exception as e:
            logger.error("Unexpected error occured while handling authorization request response: %s",
                         str(e),
                         exc_info=True)
            raise OAuthCodeFlowError(
                f"Unexpected error occurred while handling authorization request response: {str(e)}") from e

    async def _handle_oauth_302_consent_browser(self, location_header: str,
                                                authentication_provider: OAuth2Config) -> None:
        """
        Handles the consent prompt redirect for different execution environments.

        Args:
            location_header (str) : Location header from authorization server HTTP 302 consent prompt redirect.
            authentication_provider (OAuth2Config): The registered OAuth2.0 authentication provider.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.data_models.interactive import HumanPromptNotification
        try:
            if authentication_provider.consent_prompt_mode == ConsentPromptMode.BROWSER:

                default_browser = webbrowser.get()
                default_browser.open(location_header)

            if authentication_provider.consent_prompt_mode == ConsentPromptMode.FRONTEND:

                authentication_provider_name: str | None = _CredentialsManager(
                )._get_registered_authentication_provider_name(authentication_provider)

                logger.info(
                    "\n\n******************************************************************\n\n"
                    "OAuth2.0 consent request needed for Authentication provider: [ %s ] "
                    "in headless execution mode.\n"
                    "Please ensure your client sends a POST request with the registered [ consent_prompt_key ] to "
                    "continue OAuth2.0 code flow.\n\n"
                    "******************************************************************",
                    authentication_provider_name)

                if (self.message_handler):
                    await self.message_handler.create_websocket_message(
                        HumanPromptNotification(text="OAuth2.0 consent request needed for authentication provider: "
                                                f"[ {authentication_provider_name} ]. "
                                                "Navigate to the '/aiq-auth' page to continue OAuth2.0 code flow."))

                authentication_provider.consent_prompt_location_url = location_header

                await _CredentialsManager()._wait_for_consent_prompt_url()

                authentication_provider.consent_prompt_location_url = None

        except webbrowser.Error as e:
            logger.error("Unable to open defualt browser: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Unable to complete OAuth2.0 process.") from e

        except Exception as e:
            logger.error("Exception occured: %s", str(e), exc_info=True)
            raise OAuthCodeFlowError("Unable to complete OAuth2.0 process.") from e

    async def _oauth_400_status_code_handler(self, response: httpx.Response) -> None:
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
            raise OAuthCodeFlowError("Invalid request. Please check the request parameters. "
                                     f"Response code: {response.status_code}, Response description: {response.text}")

        # 401 Unauthorized: Token is missing, revoked, invlaid or expired.
        elif response.status_code == 401:
            raise OAuthCodeFlowError("Access token is missing, revoked, or expired. Please re-authenticate. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")

        # 403 Forbidden: The client is authenticated but does not have proper permission.
        elif response.status_code == 403:
            raise OAuthCodeFlowError("Access token is valid, but the client does not have permission to access the "
                                     "requested resource. Please check your permissions. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")

        # 404 Not Found: The requested endpoint or resource server does not exist.
        elif response.status_code == 404:
            raise OAuthCodeFlowError("The requested endpoint does not exist. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")

        # 405 Method Not Allowed: HTTP method not allowed to the authorization server.
        elif response.status_code == 405:
            raise OAuthCodeFlowError("The HTTP method is not allowed. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")
        else:
            raise OAuthCodeFlowError("Unknown response. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")
