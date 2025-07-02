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

import typing
from abc import ABC
from abc import abstractmethod

import httpx

from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
from aiq.data_models.authentication import ConsentPromptMode
from aiq.data_models.authentication import HeaderAuthScheme
from aiq.data_models.authentication import HTTPMethod
from aiq.data_models.authentication import OAuth2AuthorizationQueryParams
from aiq.data_models.authentication import OAuth2TokenRequest

if typing.TYPE_CHECKING:
    from aiq.authentication.response_manager import ResponseManager

AUTHORIZATION_HEADER = "Authorization"


class RequestManagerBase:
    """
    Base class for handling API requests.
    This class provides an interface for making API requests.
    """
    pass


class ResponseManagerBase:
    """
    Base class for handling API responses.
    This class provides an interface for handling API responses.
    """
    pass


class AuthenticationManagerBase(ABC):
    """
    Base class for authenticating to API services.
    This class provides an interface for authenticating to API services.
    """

    @property
    @abstractmethod
    def config_name(self) -> str | None:
        """
        Get the name of the authentication configuration.

        Returns:
            str | None: The name of the authentication configuration, or None if not set.
        """
        pass

    @config_name.setter
    @abstractmethod
    def config_name(self, config_name: str | None) -> None:
        """
        Set the name of the authentication configuration.

        Args:
            config_name (str | None): The name of the authentication configuration.
        """
        pass

    @abstractmethod
    async def send_request(self,
                           url: str,
                           http_method: str | HTTPMethod,
                           headers: dict | None = None,
                           query_params: dict | None = None,
                           body_data: dict | None = None) -> httpx.Response | None:
        """
        Makes a generic HTTP request.

        Args:
            url: The URL to send the request to
            http_method: The HTTP method to use (GET, POST, etc.)
            headers: Optional dictionary of HTTP headers
            query_params: Optional dictionary of query parameters
            body_data: Optional dictionary representing the request body

        Returns:
            httpx.Response | None: The response from the HTTP request, or None if an error occurs.
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validates the credentials for the authentication manager.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        pass

    @abstractmethod
    async def construct_authentication_header(self, header_auth_scheme: HeaderAuthScheme) -> httpx.Headers | None:
        """
        Constructs the authenticated HTTP header based on the authentication scheme.

        Args:
            header_auth_scheme: The authentication scheme to use (BEARER, BASIC, X_API_KEY, CUSTOM)

        Returns:
            httpx.Headers | None: The constructed authentication header, or None if construction fails.
        """
        pass


class OAuthClientBase(AuthenticationManagerBase, ABC):
    """
    Base class for managing OAuth clients.
    This class provides an interface for managing OAuth clients.
    """

    @property
    @abstractmethod
    def config(self) -> OAuthUserConsentConfigBase | None:
        """
        Get the OAuth authentication configuration.

        Returns:
            OAuthUserConsentConfigBase | None: The OAuth authentication configuration, or None if not set.
        """
        pass

    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validates the OAuth credentials.

        Returns:
            bool: True if credentials are valid, False otherwise.
        """
        pass

    @abstractmethod
    def _construct_authorization_query_params(self, response_type: str, prompt: str) -> OAuth2AuthorizationQueryParams:
        """
        Constructs the OAuth2 authorization query parameters for the authorization URL.

        Args:
            response_type: The OAuth2 response type (typically "code")
            prompt: The consent prompt behavior

        Returns:
            OAuth2AuthorizationQueryParams: The constructed query parameters for OAuth2 authorization
        """
        pass

    @abstractmethod
    def _construct_token_request_body(self,
                                      redirect_uri: str,
                                      authorization_code: str,
                                      grant_type: str = "authorization_code") -> OAuth2TokenRequest:
        """
        Constructs the OAuth2 token request body for exchanging authorization code for access token.

        Args:
            redirect_uri: The redirect URI used in the authorization request
            authorization_code: The authorization code received from the OAuth provider
            grant_type: The OAuth2 grant type (default: "authorization_code")

        Returns:
            OAuth2TokenRequest: The constructed token request body for OAuth2 token exchange
        """
        pass

    @property
    def response_manager(self) -> "ResponseManager | None":
        """
        Get the response manager for the authentication manager.

        Returns:
            ResponseManager | None: The response manager or None if not set.
        """
        return getattr(self, '_response_manager', None)

    @response_manager.setter
    def response_manager(self, response_manager: "ResponseManager") -> None:
        """
        Set the response manager for the authentication manager.

        Args:
            response_manager: The response manager to set
        """
        self._response_manager = response_manager

    @property
    @abstractmethod
    def consent_prompt_mode(self) -> ConsentPromptMode | None:
        """
        Get the consent prompt mode for the OAuth client.

        Returns:
            ConsentPromptMode | None: The consent prompt mode (BROWSER or FRONTEND), or None if not set.
        """
        pass

    @consent_prompt_mode.setter
    @abstractmethod
    def consent_prompt_mode(self, consent_prompt_mode: ConsentPromptMode) -> None:
        """
        Set the consent prompt mode for the OAuth client.

        Args:
            consent_prompt_mode: The consent prompt mode to set.
        """
        pass

    @abstractmethod
    async def initiate_authorization_flow_console(self) -> None:
        """
        Starts a lightweight server to initiate and complete the authorization
        flow for OAuth clients in headless (console-based) environments.
        """
        pass

    @abstractmethod
    async def shut_down_code_flow_console(self) -> None:
        """
        Shuts down the lightweight server used to complete the authorization flow
        for OAuth clients in headless (console-based) environments.
        """
        pass

    @abstractmethod
    async def initiate_authorization_flow_server(self) -> None:
        """
        Initiates and completes the authorization flow for OAuth clients
        running in server-based environments.
        """
        pass

    @abstractmethod
    async def shut_down_code_flow_server(self) -> None:
        """
        Cleans up the OAuth client authorization flow in server
        environments in the event of unexpected errors.
        """
        pass
