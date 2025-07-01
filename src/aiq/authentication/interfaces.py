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
from aiq.data_models.authentication import HTTPAuthScheme

if typing.TYPE_CHECKING:
    from aiq.authentication.response_manager import ResponseManager


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
    async def validate_credentials(self) -> bool:
        """
        Validates the credentials for the authentication manager.
        """
        pass

    @abstractmethod
    async def construct_authentication_header(self, http_auth_scheme: HTTPAuthScheme) -> httpx.Headers | None:
        """
        Constructs the authenticated HTTP header based on the authentication scheme.

        Applies to:
        - Basic
        - Bearer
        - Digest
        - OAuth2
        - OpenID Connect
        - Custom (if header-based)
        """
        pass

    @abstractmethod
    async def construct_authentication_query(self, http_auth_scheme: HTTPAuthScheme) -> httpx.QueryParams | None:
        """
        Constructs the authenticated HTTP query based on the authentication scheme.

        Applies to:
        - Custom (if using query parameters)
        - API key (query, if used)
        """
        pass

    @abstractmethod
    async def construct_authentication_cookie(self, http_auth_scheme: HTTPAuthScheme) -> httpx.Cookies | None:
        """
        Constructs the authenticated HTTP cookie based on the authentication scheme.

        Applies to:
        - Cookie
        - Custom (if using cookies)
        """
        pass

    @abstractmethod
    async def construct_authentication_body(self, http_auth_scheme: HTTPAuthScheme) -> dict[str, typing.Any] | None:
        """
        Constructs the authenticated HTTP body based on the authentication scheme.

        Applies to:
        - Custom (rarely used schemes or special configurations)
        """
        pass

    @abstractmethod
    async def construct_authentication_custom(self, http_auth_scheme: HTTPAuthScheme) -> typing.Any | None:
        """
        Constructs the authenticated HTTP custom logic based on the authentication scheme.

        Applies to:
        - Custom only (fully user-defined logic)
        """
        pass


class OAuthClientBase(AuthenticationManagerBase, ABC):
    """
    Base class for managing OAuth clients.
    This class provides an interface for managing OAuth clients.
    """

    @abstractmethod
    async def validate_credentials(self) -> bool:
        pass

    @property
    def response_manager(self) -> "ResponseManager | None":
        """
        Get the response manager for the authentication manager.

        Returns:
            ResponseManager | None: The response manager or None if not set.
        """
        return self._response_manager

    @response_manager.setter
    def response_manager(self, response_manager: "ResponseManager") -> None:
        """
        Set the response manager for the authentication manager.
        """
        self._response_manager = response_manager

    @property
    @abstractmethod
    def consent_prompt_mode(self) -> ConsentPromptMode | None:
        """
        Get the consent prompt mode for the OAuth client.

        Returns:
            ConsentPromptMode: The consent prompt mode (BROWSER or FRONTEND).
        """
        pass

    @consent_prompt_mode.setter
    @abstractmethod
    def consent_prompt_mode(self, consent_prompt_mode: ConsentPromptMode) -> None:
        """
        Set the consent prompt mode for the OAuth client.

        Args:
            consent_prompt_mode (ConsentPromptMode): The consent prompt mode to set.
        """
        pass

    @property
    @abstractmethod
    def config(self) -> OAuthUserConsentConfigBase | None:
        """
        Get the authentication configuration.

        Returns:
            str | None: The authentication configuration, or None if not set.
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
