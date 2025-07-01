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

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantFlowError
from aiq.authentication.interfaces import OAuthClientBase
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
                                                    oauth_client_manager: OAuthClientBase) -> None:
        """
        Handles various Auth Code Grant Flow flow responses.

        Args:
            response (httpx.Response): The HTTP response from the authentication server.
            encrypted_authentication_config (OAuthClientBase): The registered Auth Code Grant flow config.
        """
        try:
            # Handles the 302 redirect status code from Auth Code Grant flow authorization server.
            if response.status_code == 302:
                redirect_location_header: str | None = response.headers.get("Location")

                if not redirect_location_header:
                    error_message = "Missing 'Location' header in 302 response to redirect user to consent browser"
                    raise AuthCodeGrantFlowError('location_header_missing', error_message)

                await self._handle_auth_code_grant_302_consent_browser(redirect_location_header, oauth_client_manager)

            # Handles the 4xx client status codes from Auth Code Grant flow authorization server.
            elif response.status_code >= 400 and response.status_code < 500:
                await self._oauth_400_status_code_handler(response)

            elif response.status_code >= 500 and response.status_code < 600:
                await self._general_500_status_code_handler(response)

            else:
                error_message = f"Unknown response code: {response.status_code}. Response: {response.text}"
                raise AuthCodeGrantFlowError('unknown_response_code', error_message)

        except Exception as e:
            error_message = f"Unexpected error occurred while handling authorization request response: {str(e)}"
            logger.error(error_message, exc_info=True)
            raise AuthCodeGrantFlowError('auth_response_handler_failed', error_message) from e

    async def _handle_auth_code_grant_302_consent_browser(self,
                                                          location_header: str,
                                                          oauth_client_manager: OAuthClientBase) -> None:
        """
        Handles the consent prompt redirect for different execution environments.

        Args:
            location_header (str) : Location header from authorization server HTTP 302 consent prompt redirect.
            encrypted_authentication_config (OAuthClientBase): The registered Auth Code Grant flow config.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager
        from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
        from aiq.data_models.interactive import HumanPromptNotification

        oauth_config: OAuthUserConsentConfigBase | None = oauth_client_manager.config
        try:
            if oauth_client_manager.consent_prompt_mode == ConsentPromptMode.BROWSER:

                default_browser = webbrowser.get()
                default_browser.open(location_header)

            if oauth_client_manager.consent_prompt_mode == ConsentPromptMode.FRONTEND:

                authentication_config_name: str | None = oauth_client_manager.config_name

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

                if oauth_config:
                    oauth_config.consent_prompt_location_url = location_header

                    await _CredentialsManager().wait_for_consent_prompt_url()

                    oauth_config.consent_prompt_location_url = None

        except webbrowser.Error as e:
            error_message = f"Unable to complete Auth Code Grant flow process - browser open failed: {str(e)}"
            logger.error(error_message, exc_info=True)
            raise AuthCodeGrantFlowError('browser_open_failed', error_message) from e

        except Exception as e:
            error_message = (f"Unable to complete Auth Code Grant flow process - "
                             f"consent browser handling failed: {str(e)}")
            logger.error(error_message, exc_info=True)
            raise AuthCodeGrantFlowError('consent_browser_failed', error_message) from e

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
            error_message = (f"Invalid request. Please check the request parameters. "
                             f"Response code: {response.status_code}, Response description: {response.text}")
            raise AuthCodeGrantFlowError('http_400_bad_request', error_message)

        # 401 Unauthorized: Token is missing, revoked, invlaid or expired.
        elif response.status_code == 401:
            error_message = (f"Access token is missing, revoked, or expired. Please re-authenticate. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_401_unauthorized', error_message)

        # 403 Forbidden: The client is authenticated but does not have proper permission.
        elif response.status_code == 403:
            error_message = (f"Access token is valid, but the client does not have permission to access the "
                             f"requested resource. Please check your permissions. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_403_forbidden', error_message)

        # 404 Not Found: The requested endpoint or resource server does not exist.
        elif response.status_code == 404:
            error_message = (f"The requested endpoint does not exist. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_404_not_found', error_message)

        # 405 Method Not Allowed: HTTP method not allowed to the authorization server.
        elif response.status_code == 405:
            error_message = (f"The HTTP method is not allowed. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_405_method_not_allowed', error_message)

        # 422 Unprocessable Entity: The request was well-formed but contains semantic errors.
        elif response.status_code == 422:
            error_message = (f"The request was well-formed but could not be processed. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_422_unprocessable_entity', error_message)

        # 429 Too Many Requests: The client has sent too many requests in a given amount of time.
        elif response.status_code == 429:
            error_message = (f"Too many requests - you are being rate-limited. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_429_too_many_requests', error_message)
        else:
            error_message = (f"Unknown response. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_unknown_error', error_message)

    async def _general_500_status_code_handler(self, response: httpx.Response) -> None:
        """

        Handles HTTP status codes in the response and raises descriptive exceptions for
        known protocol-level and server-side error conditions.

        Args:
            response (httpx.Response): The http response.
        """
        # 500 Internal Server Error: Generic server error.
        if response.status_code == 500:
            error_message = (f"The server encountered an internal error. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_500_internal_server_error', error_message)

        # 502 Bad Gateway: Invalid response from an upstream server.
        elif response.status_code == 502:
            error_message = (f"Bad gateway - received invalid response from upstream server. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_502_bad_gateway', error_message)

        # 503 Service Unavailable: Server is currently unable to handle the request.
        elif response.status_code == 503:
            error_message = (f"Service unavailable - server cannot handle the request right now. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_503_service_unavailable', error_message)

        # 504 Gateway Timeout: The server did not receive a timely response from an upstream server.
        elif response.status_code == 504:
            error_message = (f"Gateway timeout - the server did not receive a timely response. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_504_gateway_timeout', error_message)
        else:
            error_message = (f"Unknown response. "
                             f"Response code: {response.status_code}, Response Description: {response.text}")
            raise AuthCodeGrantFlowError('http_unknown_error', error_message)
