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

from aiq.authentication.exceptions import AuthCodeGrantError
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.data_models.authentication import ConsentPromptMode
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

    async def handle_auth_code_grant_response_codes(self,
                                                    response: httpx.Response,
                                                    authentication_config: AuthCodeGrantConfig) -> None:
        """
        Handles various Auth Code Grant Flow flow responses.

        Args:
            response (httpx.Response): The HTTP response from the authentication server.
            authentication_config (AuthCodeGrantConfig): The registered Auth Code Grant flow config.
        """
        try:
            # Handles the 302 redirect status code from Auth Code Grant flow authorization server.
            if response.status_code == 302:
                redirect_location_header: str | None = response.headers.get("Location")

                if not redirect_location_header:
                    raise AuthCodeGrantError(
                        "Missing 'Location' header in 302 response to redirect user to consent browser.")

                await self._handle_auth_code_grant_302_consent_browser(redirect_location_header, authentication_config)

            # Handles the 4xx client status codes from Auth Code Grant flow authorization server.
            elif response.status_code >= 400 and response.status_code < 500:
                await self._oauth_400_status_code_handler(response)

            else:
                raise AuthCodeGrantError(f"Unknown response code: {response.status_code}. Response: {response.text}")

        except Exception as e:
            logger.error("Unexpected error occured while handling authorization request response: %s",
                         str(e),
                         exc_info=True)
            raise AuthCodeGrantError(
                f"Unexpected error occurred while handling authorization request response: {str(e)}") from e

    async def _handle_auth_code_grant_302_consent_browser(self,
                                                          location_header: str,
                                                          authentication_config: AuthCodeGrantConfig) -> None:
        """
        Handles the consent prompt redirect for different execution environments.

        Args:
            location_header (str) : Location header from authorization server HTTP 302 consent prompt redirect.
            authentication_config (AuthCodeGrantConfig): The registered Auth Code Grant flow config.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.data_models.interactive import HumanPromptNotification
        try:
            if authentication_config.consent_prompt_mode == ConsentPromptMode.BROWSER:

                default_browser = webbrowser.get()
                default_browser.open(location_header)

            if authentication_config.consent_prompt_mode == ConsentPromptMode.FRONTEND:

                authentication_config_name: str | None = _CredentialsManager(
                ).get_registered_authentication_config_name(authentication_config)

                logger.info(
                    "\n\n******************************************************************\n\n"
                    "Authorization Code Grant flow consent request needed for Authentication config: [ %s ] "
                    "in headless execution mode.\n"
                    "Please ensure your client sends a POST request with the registered [ consent_prompt_key ] to "
                    "continue Auth Code Grant flow.\n\n"
                    "******************************************************************",
                    authentication_config_name)

                if (self.message_handler):
                    await self.message_handler.create_websocket_message(
                        HumanPromptNotification(
                            text="Auth Code Grant flow consent request needed for authentication config: "
                            f"[ {authentication_config_name} ]. "
                            "Navigate to the '/aiq-auth' page to continue Auth Code Grant flow."))

                authentication_config.consent_prompt_location_url = location_header

                await _CredentialsManager().wait_for_consent_prompt_url()

                authentication_config.consent_prompt_location_url = None

        except webbrowser.Error as e:
            logger.error("Unable to open defualt browser: %s", str(e), exc_info=True)
            raise AuthCodeGrantError("Unable to complete Auth Code Grant flow process.") from e

        except Exception as e:
            logger.error("Exception occured: %s", str(e), exc_info=True)
            raise AuthCodeGrantError("Unable to complete Auth Code Grant flow  process.") from e

    async def _oauth_400_status_code_handler(self, response: httpx.Response) -> None:
        """
        Handles the response to a protected resource request with an authentication attempt using
        an expired or invalid access token. According to RFC 6750 When a request fails, the resource server
        responds using the appropriate HTTP status code (typically, 400, 401, 403, or 405) and
        includes one of the following error codes in the response: invalid_request, invalid_token,
        insufficient_scope.

        Args:
            response (httpx.Response): The response form the Auth Code Grant flow authentication server.
        """
        # 400 Bad Request: Invalid refresh token provided or malformed request.
        if response.status_code == 400:
            raise AuthCodeGrantError("Invalid request. Please check the request parameters. "
                                     f"Response code: {response.status_code}, Response description: {response.text}")

        # 401 Unauthorized: Token is missing, revoked, invlaid or expired.
        elif response.status_code == 401:
            raise AuthCodeGrantError("Access token is missing, revoked, or expired. Please re-authenticate. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")

        # 403 Forbidden: The client is authenticated but does not have proper permission.
        elif response.status_code == 403:
            raise AuthCodeGrantError("Access token is valid, but the client does not have permission to access the "
                                     "requested resource. Please check your permissions. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")

        # 404 Not Found: The requested endpoint or resource server does not exist.
        elif response.status_code == 404:
            raise AuthCodeGrantError("The requested endpoint does not exist. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")

        # 405 Method Not Allowed: HTTP method not allowed to the authorization server.
        elif response.status_code == 405:
            raise AuthCodeGrantError("The HTTP method is not allowed. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")
        else:
            raise AuthCodeGrantError("Unknown response. "
                                     f"Response code: {response.status_code}, Response Description: {response.text}")
