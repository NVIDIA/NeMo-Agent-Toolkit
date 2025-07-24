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

from abc import ABC
from abc import abstractmethod

import httpx

from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
from aiq.data_models.authentication import AuthenticationBaseConfig, AuthenticatedContext, AuthFlowType, AuthResult
from aiq.data_models.authentication import CredentialLocation
from aiq.data_models.authentication import HeaderAuthScheme
from aiq.data_models.authentication import OAuth2AuthorizationQueryParams
from aiq.data_models.authentication import OAuth2TokenRequest
from aiq.front_ends.fastapi.message_handler import MessageHandler

AUTHORIZATION_HEADER = "Authorization"


class AuthenticationClientBase(ABC):
    """
    Base class for authenticating to API services.
    This class provides an interface for authenticating to API services.
    """

    def __init__(self, config: AuthenticationBaseConfig):
        """
        Initialize the AuthenticationClientBase with the given configuration.

        Args:
            config (AuthenticationBaseConfig): Configuration items for authentication.
        """
        self.config = config

    @abstractmethod
    async def authenticate(self, user_id: str) -> AuthResult:
        """
        Perform the authentication process for the client.

        This method handles the necessary steps to authenticate the client with the
        target API service, which may include obtaining tokens, refreshing credentials,
        or completing multi-step authentication flows.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        ## This method will call the frontend FlowHandlerBase `authenticate` method
        pass


class FlowHandlerBase(ABC):
    """
    Handles front-end specifc flows for authentication clients.

    Each front end will define a FlowHandler that will implement the authenticate method.

    The `authenticate` method will be stored as the callback in the AIQContextState.user_auth_callback
    """

    @staticmethod
    async def authenticate(config: AuthenticationBaseConfig, method: AuthFlowType) -> AuthenticatedContext:
        """
        Perform the authentication process for the client.

        This method handles the necessary steps to authenticate the client with the
        target API service, which may include obtaining tokens, refreshing credentials,
        or completing multistep authentication flows.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass


#######################################################################
## The below interface needs to be updated. OAuth is broken currently##
#######################################################################

class OAuthClientBase(AuthenticationClientBase, ABC):
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

    async def authenticate(self, user_id: str) -> AuthResult:
        """
        Perform the authentication process for the client.

        This method handles the necessary steps to authenticate the client with the
        target API service, which may include obtaining tokens, refreshing credentials,
        or completing multi-step authentication flows.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        ## This method will call the frontend FlowHandlerBase `authenticate` method
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
    def construct_authorization_query_params(self, response_type: str, prompt: str) -> OAuth2AuthorizationQueryParams:
        """
        Constructs the query parameters used in the OAuth 2.0 authorization request URL.

        This method generates the required query parameters for initiating an authorization
        request to the authorization server, applicable to flows such as Authorization Code,
        Authorization Code with PKCE, Hybrid, and other redirect-based OAuth 2.0 flows.

        Args:
            response_type (str): The OAuth 2.0 response type indicating the expected authorization response
            (e.g., "code", "token", or "id_token").
            prompt (str): The type of user interaction or consent prompt requested (e.g., "consent", "login", "none").

        Returns:
            OAuth2AuthorizationQueryParams: A model containing the full set of query parameters to
                                            be included in the authorization request URL.
        """
        pass

    @abstractmethod
    async def send_authorization_request(
            self, authorization_url: str,
            authorization_query_params: OAuth2AuthorizationQueryParams) -> httpx.Response | None:
        """
        Sends the OAuth 2.0 authorization request to the authorization server.

        This method builds the full authorization request URL by combining the base authorization URL
        with the provided query parameters. It then initiates the authorization request, which may
        trigger a user consent prompt depending on the flow (e.g., Authorization Code Grant, Device Flow).

        Args:
            authorization_url (str): The base URL to the authorization server's authorization endpoint.
            authorization_query_params (OAuth2AuthorizationQueryParams): Query parameters such as
            client_id, redirect_uri, response_type, scope, and state to be merged into the URL.

        Returns:
            httpx.Response | None: The HTTP response received from the authorization server, or None
            if user interaction is required through a browser or external interface.
        """
        pass

    @abstractmethod
    async def handle_authorization_server_response(self, response: httpx.Response | None) -> None:
        """
        Handles server responses returned during the OAuth 2.0 authorization process.
        This method interprets and reacts to various response codes and payloads from
        the authorization server, determining whether the flow should continue, retry,
        or fail. It is designed to be applicable across OAuth 2.0 flows that need
        user consent.

        Args:
            response (httpx.Response): The HTTP response received after redirecting the user to the authorization
            endpoint and receiving a callback.
        """
        pass

    @abstractmethod
    async def send_token_request(self,
                                 client_authorization_path: str,
                                 client_authorization_endpoint: str,
                                 authorization_code: str) -> httpx.Response | None:
        """
        Sends a token request to the token endpoint of the authorization server. This method handles the final step in
        OAuth 2.0 flows that require exchanging an authorization code or grant credential for an access token. It
        constructs and submits a request to the token endpoint, using the required parameters and headers defined by the
        flow (e.g., client credentials, redirect URI, code verifier).

        Args:
            client_authorization_path (str): The relative path on the client used to trigger token requests.
            client_authorization_endpoint (str): The full URL to the authorization server's token endpoint.
            authorization_code (str): The code or credential issued by the authorization server.

        Returns:
            httpx.Response | None: The response from the token endpoint, or None if the request failed.
        """
        pass

    @abstractmethod
    async def process_token_response(self, response: httpx.Response) -> None:
        """
        Processes the HTTP response received from the OAuth 2.0 token endpoint.

        This method is responsible for extracting and validating the token response returned
        by the authorization server, including access tokens and any optional fields such as
        refresh tokens, ID tokens, token type, expiration time, and granted scopes.
        The extracted values are typically delegated to a token storage handler.

        Args:
            response (httpx.Response): The HTTP response object returned from the token endpoint.

        Returns:
            None: This method does not return a value.
        """
        pass

    @abstractmethod
    def construct_token_request_body(self,
                                     redirect_uri: str,
                                     authorization_code: str,
                                     grant_type: str = "authorization_code") -> OAuth2TokenRequest:
        """
        Constructs the OAuth2 token request body for exchanging authorization code for access token.

        Args:
            redirect_uri (str): The redirect URI used in the authorization request
            authorization_code (str): The authorization code received from the OAuth provider
            grant_type (str): The OAuth2 grant type (default: "authorization_code")
            - `authorization_code`: (Authorization Code Grant, Authorization Code with PKCE)
            - `client_credentials`: (Client Credentials Grant)
            - `urn:ietf:params:oauth:grant-type:device_code`: (Device Code Grant)
            - `refresh_token`: (Refresh Token Grant)

        Returns:
            OAuth2TokenRequest: The constructed token request body for OAuth2 token exchange
        """
        pass


class RequestManagerBase:
    """
    Base class for handling API requests.
    This class provides an interface for making API requests.
    """
    pass


class ResponseManagerBase(ABC):
    """
    Base class for handling API responses.
    This class provides an interface for handling API responses.
    """

    @property
    @abstractmethod
    def message_handler(self) -> "MessageHandler | None":
        """
        Get the message handler for the response manager.
        """
        pass

    @message_handler.setter
    @abstractmethod
    def message_handler(self, message_handler: "MessageHandler") -> None:
        """
        Set the message handler for the response manager.
        """
        pass

    @abstractmethod
    async def handle_consent_prompt_redirect(self, location_header: str) -> None:
        """
        Handles redirect-based consent prompts initiated by OAuth 2.0 flows.

        This method is responsible for processing the redirect URI received from the
        authorization server (typically a 302 Location header). Depending on the execution
        environment (e.g., headless server, local development, desktop), this may involve
        opening a browser, triggering a frontend event, or notifying the user to complete
        the consent flow manually.

        Args:
            location_header (str): The redirect URL containing the consent prompt.

        Raises:
            NotImplementedError: Must be implemented by subclasses tailored to the execution context.
        """
        pass