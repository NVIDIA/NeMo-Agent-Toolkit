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

import json
import logging
import re
import urllib.parse

import httpx
from pydantic import ValidationError

from aiq.authentication.authentication_manager import AuthenticationManager
from aiq.authentication.exceptions import BaseUrlValidationError
from aiq.authentication.exceptions import BodyValidationError
from aiq.authentication.exceptions import HeaderValidationError
from aiq.authentication.exceptions import OAuthError
from aiq.authentication.exceptions import QueryParameterValidationError
from aiq.authentication.interfaces import RequestManagerBase
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import AuthenticationEndpoint
from aiq.data_models.authentication import HTTPMethod
from aiq.data_models.authentication import OAuth2AuthQueryParams
from aiq.data_models.authentication import OAuth2Config

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RequestManager(RequestManagerBase):

    def __init__(self) -> None:
        self._authentication_manager: AuthenticationManager = AuthenticationManager(self)
        self._response_manager: ResponseManager = ResponseManager()

    @property
    def authentication_manager(self) -> AuthenticationManager:
        return self._authentication_manager

    def _validate_data(self, input_dict: dict) -> None:
        # Check to ensure all parameters have values
        invalid_values = [key for key, value in input_dict.items() if not value]

        if invalid_values:
            raise ValueError(f"Empty or invalid values for input: {input_dict}. "
                             f"Invalid Values: {', '.join(invalid_values)}")

    def _validate_base_url(self, url: str | httpx.URL) -> None:
        """Validates URL and Raises BaseUrlError if the URL is not a valid URL."""

        if isinstance(url, httpx.URL):
            url = str(url)

        parsed_url: urllib.parse.ParseResult = urllib.parse.urlparse(url)  # TODO EE: Add Tests.

        # Ensure URL has both scheme and network location
        if not parsed_url.scheme or not parsed_url.netloc:
            raise BaseUrlValidationError("URL must have both scheme and network location.")

        # Ensure URL scheme is (http or https)
        if parsed_url.scheme not in ['http', 'https']:
            raise BaseUrlValidationError(f"Unsupported URL scheme: {parsed_url.scheme}. Must be http or https.")

        # Ensure URL starts with a '/'
        if not parsed_url.path.startswith("/"):
            raise BaseUrlValidationError("URL path should start with '/'")

    def _validate_http_method(self, http_method: str | HTTPMethod) -> None:
        """
        Validates that the provided HTTP method is one of the allowed standard methods.

        Args:
            http_method (str): The HTTP method to validate (e.g., 'GET', 'POST').
        """
        try:
            HTTPMethod(http_method.upper())  # TODO EE: Add Tests
        except ValueError as e:
            valid_http_methods = ', '.join([method.value for method in HTTPMethod])
            raise ValueError(f"Invalid HTTP method: '{http_method}'. Must be one of {valid_http_methods}.") from e

    def _validate_headers(self, headers: dict | httpx.Headers | None) -> None:
        """
        Validates that the provided headers are valid for HTTP request.

        Args:
            headers (dict): Dictionary of headers.
        """
        if headers is None:
            return

        if isinstance(headers, httpx.Headers):
            headers = dict(headers)

        self._validate_data(headers)

        for key, value in headers.items():
            # Checking for valid ascii characters.
            if not re.fullmatch(r"[A-Za-z0-9-]+", key):  # TODO EE: Add Tests
                raise HeaderValidationError(f"Invalid header name: {key}")

            # Checking for any disallowed control characters.
            if any(ord(char) < 32 and char != '\t' or ord(char) == 127 for char in str(value)):
                raise HeaderValidationError(f"Invalid control character in header value: {key}: {value}")

    def _validate_query_parameters(self, query_params: dict | httpx.QueryParams | None) -> None:
        """
        Validates that the provided query parameters are valid for HTTP request.

        Args:
            query_params (dict, httpx.QueryParams): Dictionary of query parameters.
        """
        if query_params is None:
            return

        if isinstance(query_params, httpx.QueryParams):
            query_params = dict(query_params)

        self._validate_data(query_params)

        # Checking if the key can be safely encoded
        for key, value in query_params.items():  # TODO EE: Add Tests
            try:
                urllib.parse.quote(str(key))
                urllib.parse.quote(str(value))
            except Exception as e:
                raise QueryParameterValidationError(
                    f"Unable to encode query parameters safely: ({key}: {value})") from e

    def _validate_body_data(self, body_data: dict | None) -> None:
        """
        Validates that the provided body is valid for HTTP request.

        Args:
            body_data (dict): Dictionary representing HTTP body.
        """
        if body_data is None:
            return
        try:
            json.dumps(body_data)
        except (TypeError, ValueError) as e:
            raise BodyValidationError(f"Request body is not JSON serializable: {e}") from e

    async def _build_oauth_authorization_url(self,
                                             authentication_provider: OAuth2Config,
                                             response_type: str = "code",
                                             prompt: str = "consent") -> httpx.URL:
        """
        Construct an authorization URL to initiate the OAuth2.0 Code Flow.

        Args:
            authentication_provider (OAuth2Config): The registered authentication provider.
        """
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

        try:
            full_authorization_url: httpx.URL = None

            # Validate authorization url.
            self._validate_base_url(authentication_provider.authorization_url)

            # Construct OAuth2.0 query parameters.
            query_params: OAuth2AuthQueryParams = OAuth2AuthQueryParams(
                audience=authentication_provider.audience,
                client_id=authentication_provider.client_id,
                state=authentication_provider.state,
                scope=(" ".join(authentication_provider.scope)),
                redirect_uri=(f"{authentication_provider.client_server_url}"
                              f"{FastApiFrontEndConfig().authorization.path}"
                              f"{AuthenticationEndpoint.REDIRECT_URI.value}"),
                response_type=response_type,
                prompt=prompt)

            self._validate_query_parameters(query_params.model_dump())

            full_authorization_url = httpx.URL(authentication_provider.authorization_url).copy_merge_params(
                query_params.model_dump())

        except (BaseUrlValidationError, QueryParameterValidationError, ValueError, ValidationError, Exception) as e:
            logger.error("An error occured while building authorization url: %s", str(e), exc_info=True)
            raise OAuthError("An error occured while building authorization url.") from e

        return full_authorization_url

    async def send_oauth_authorization_request(self, authentication_provider: OAuth2Config):
        """
        Constructs OAuth2.0 Code Flow Authoriation URL and sends request to authentication server.

        Args:
            authentication_provider (OAuth2Config): The registered OAuth2.0 provider
        """
        try:
            authorization_url: httpx.URL = await self._build_oauth_authorization_url(authentication_provider)

            response: httpx.Response | None = await self.send_request(url=authorization_url, http_method="GET")

            await self._response_manager._handle_oauth_authorization_response(response, authentication_provider)

        except Exception as e:
            logger.error("Unexpected error occured during authorization request process: %s", str(e), exc_info=True)
            raise OAuthError("Unexpected error occured during authorization request process:") from e

    async def send_request(self,
                           url: str,
                           http_method: str | HTTPMethod,
                           authentication_provider: str | None = None,
                           headers: dict | None = None,
                           query_params: dict | None = None,
                           data: dict | None = None) -> httpx.Response | None:
        """
        Makes an arbitrary HTTP request.

        Args:
            url (str | httpx.URL): The base URL to which the request will be sent.
            http_method (str | HTTPMethod): The HTTP method to use for the request (e.g., "GET", "POST").
            headers (dict | None): Optional dictionary of HTTP headers.
            query_params (dict | None): Optional dictionary of query parameters.
            data (dict | None): Optional dictionary representing the request body.
        """
        try:
            authentication_header: httpx.Headers | None = await self._get_authenticated_header(authentication_provider)
            headers: httpx.Headers = httpx.Headers({**(headers or {}), **(authentication_header or {})})

            # Validate the incoming base url.
            self._validate_base_url(url)

            # Validate the incoming http method.
            self._validate_http_method(http_method)

            # Validate incoming header parameters.
            self._validate_headers(headers)

            # Validate incoming query parameters.
            self._validate_query_parameters(query_params)

            # Validate incoming body
            self._validate_body_data(data)

            response: httpx.Response | None = None

            async with httpx.AsyncClient() as client:
                if http_method.upper() == HTTPMethod.GET.value:
                    response = await client.get(url, params=query_params, headers=headers, timeout=10.0)
                if http_method.upper() == HTTPMethod.POST.value:
                    response = await client.post(url, params=query_params, headers=headers, json=data, timeout=10.0)
                if http_method.upper() == HTTPMethod.PUT.value:
                    response = await client.put(url, params=query_params, headers=headers, json=data, timeout=10.0)
                if http_method.upper() == HTTPMethod.DELETE.value:
                    response = await client.delete(url, params=query_params, headers=headers, timeout=10.0)

        except (BaseUrlValidationError,
                ValidationError,
                HeaderValidationError,
                QueryParameterValidationError,
                BodyValidationError) as e:
            logger.error("An error occured while building request url: %s", str(e), exc_info=True)
            return None

        except (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as e:

            logger.error("An error occured while sending request: %s", str(e), exc_info=True)
            return None

        except Exception as e:
            logger.error("Unexpected eror occured sending request %s", str(e), exc_info=True)
            return None

        return response

    async def _get_authenticated_header(self, authentication_provider: str | None = None) -> httpx.Headers | None:
        """
        Gets the authenticated header for the registered authentication provider.

        Args:
            authentication_provider (str | None): The name of the registered authentication provider.

        Returns:
            httpx.Headers | None: _description_ #TODO EE: Check all doc strings
        """

        # If no authentication provider is passed, no authentication header is required, return None.
        if authentication_provider is None:
            return None

        # Ensure authentication provider credentials are valid and functional.
        is_validated: bool = await self.authentication_manager._validate_auth_provider_credentials(
            authentication_provider)

        if (is_validated):
            return await self.authentication_manager._construct_authentication_header(authentication_provider)
        else:
            get_auth_header = await self._authentication_manager._set_auth_provider_credentials(authentication_provider)

            if (get_auth_header):
                return await self.authentication_manager._construct_authentication_header(authentication_provider)

        return None
