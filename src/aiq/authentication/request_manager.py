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

from aiq.authentication.exceptions import APIRequestError
from aiq.authentication.exceptions import AuthCodeGrantError
from aiq.authentication.exceptions import BaseUrlValidationError
from aiq.authentication.exceptions import BodyValidationError
from aiq.authentication.exceptions import HTTPHeaderValidationError
from aiq.authentication.exceptions import HTTPMethodValidationError
from aiq.authentication.exceptions import QueryParameterValidationError
from aiq.authentication.interfaces import RequestManagerBase
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import AuthCodeGrantQueryParams
from aiq.data_models.authentication import AuthenticationEndpoint
from aiq.data_models.authentication import HTTPMethod

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RequestManager(RequestManagerBase):

    def __init__(self) -> None:
        self._response_manager: ResponseManager = ResponseManager()

    @property
    def response_manager(self) -> ResponseManager:
        return self._response_manager

    def validate_data(self, input_dict: dict) -> None:
        """
        Validates that the provided dictionary has valid keys and values.

        Args:
            input_dict (dict): The dictionary of key-value pairs to validate.

        Raises:
            ValueError: If any key is invalid or any value is empty or invalid.
        """
        # Check empty, whitespace-only, or non-string keys.
        invalid_keys = [
            repr(key) for key in input_dict.keys() if key is None or not isinstance(key, str) or key.strip() == ""
        ]

        if invalid_keys:
            raise ValueError(f"Invalid keys detected in input: {input_dict}. "
                             f"Invalid Keys: {', '.join(invalid_keys)}")

        # Check for None or empty-string values.
        invalid_values = [
            key for key, value in input_dict.items()
            if value is None or (isinstance(value, str) and value.strip() == "")
        ]
        if invalid_values:
            raise ValueError(f"Empty or invalid values detected in input: {input_dict}. "
                             f"Invalid Values: {', '.join(invalid_values)}")

    def validate_base_url(self, url: str | httpx.URL) -> None:
        """Validates URL and Raises BaseUrlError if the URL is not a valid URL."""

        if isinstance(url, httpx.URL):
            url = str(url)

        parsed_url: urllib.parse.ParseResult = urllib.parse.urlparse(url)

        # Ensure URL has both scheme and network location
        if not parsed_url.scheme or not parsed_url.netloc:
            raise BaseUrlValidationError("URL must have both scheme and network location.")

        # Ensure URL scheme is (http or https)
        if parsed_url.scheme not in ['http', 'https']:
            raise BaseUrlValidationError(f"Unsupported URL scheme: {parsed_url.scheme}. Must be http or https.")

        # Ensure URL starts with a '/'
        if not parsed_url.path.startswith("/"):
            raise BaseUrlValidationError("URL path should start with '/'")

    def validate_http_method(self, http_method: str | HTTPMethod) -> None:
        """
        Validates that the provided HTTP method is one of the allowed standard methods.

        Args:
            http_method (str): The HTTP method to validate (e.g., 'GET', 'POST').
        """
        try:
            HTTPMethod(http_method.upper())
        except ValueError as e:
            valid_http_methods = ', '.join([method.value for method in HTTPMethod])
            raise HTTPMethodValidationError(
                f"Invalid HTTP method: '{http_method}'. Must be one of {valid_http_methods}.") from e

    def validate_headers(self, headers: dict | httpx.Headers | None) -> None:
        """
        Validates that the provided headers are valid for an HTTP request.

        Args:
            headers (dict): Dictionary of headers.
        """
        try:
            if headers is None:
                return None

            if isinstance(headers, httpx.Headers):
                headers = dict(headers)

            self.validate_data(headers)

            for key, value in headers.items():

                # Checking for valid ASCII characters in the header name
                if not re.fullmatch(r"[A-Za-z0-9-]+", key):
                    raise HTTPHeaderValidationError(f"Invalid header name: {key}")

                # Checking for disallowed control characters
                if any(ord(char) < 32 and char != '\t' or ord(char) == 127 for char in value):
                    raise HTTPHeaderValidationError(f"Invalid control character in header value: {key}: {value}")

        except ValueError as e:
            raise HTTPHeaderValidationError(f"Invalid header data: {e}") from e

    def validate_query_parameters(self, query_params: dict | httpx.QueryParams | None) -> None:
        """
        Validates that the provided query parameters are valid for an HTTP request.

        Args:
            query_params (dict, httpx.QueryParams): Dictionary of query parameters.
        """
        try:
            if query_params is None:
                return None

            if isinstance(query_params, httpx.QueryParams):
                query_params = dict(query_params)

            self.validate_data(query_params)

            for key, value in query_params.items():

                # Catch keys with leading/trailing whitespace to prevent ambiguous parsing or bypassing
                if key.strip() != key:
                    raise QueryParameterValidationError(f"Key has leading or trailing whitespace: '{key}'")

                # Catch newlines in keys to prevent header injection and log splitting vulnerabilities
                if isinstance(key, str) and ('\n' in key or '\r' in key):
                    raise QueryParameterValidationError(f"Key contains newline or control character: '{key}'")

                # Catch newlines in values to avoid header injection and log splitting vulnerabilities
                if isinstance(value, str) and ('\n' in value or '\r' in value):
                    raise QueryParameterValidationError(
                        f"Value contains newline or control character for key '{key}': '{value}'")

                # Try to URL-encode the key and value to ensure they are safe
                try:

                    urllib.parse.quote(str(key), safe='')
                    urllib.parse.quote(str(value), safe='')

                except Exception as e:
                    raise QueryParameterValidationError(
                        f"Unable to safely encode query parameter: ({key}: {value})") from e

        except ValueError as e:
            raise QueryParameterValidationError(f"Invalid query parameter data: {e}") from e

    def validate_body_data(self, body_data: dict | None) -> None:
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

    async def _build_auth_code_grant_url(self,
                                         authentication_config: AuthCodeGrantConfig,
                                         response_type: str = "code",
                                         prompt: str = "consent") -> httpx.URL:
        """
        Construct an authorization URL to initiate the Auth Code Grant Flow.

        Args:
            authentication_config (AuthCodeGrantConfig): The registered authentication config.
        """
        from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

        try:
            full_authorization_url: httpx.URL = None

            # Validate authorization url.
            self.validate_base_url(authentication_config.authorization_url)

            # Construct Auth Code Grant flow query parameters.
            query_params: AuthCodeGrantQueryParams = AuthCodeGrantQueryParams(
                audience=authentication_config.audience,
                client_id=authentication_config.client_id,
                state=authentication_config.state,
                scope=(" ".join(authentication_config.scope)),
                redirect_uri=(f"{authentication_config.client_server_url}"
                              f"{FastApiFrontEndConfig().authorization.path}"
                              f"{AuthenticationEndpoint.REDIRECT_URI.value}"),
                response_type=response_type,
                prompt=prompt)

            self.validate_query_parameters(query_params.model_dump())

            full_authorization_url = httpx.URL(authentication_config.authorization_url).copy_merge_params(
                query_params.model_dump())

        except (BaseUrlValidationError, QueryParameterValidationError, ValueError, ValidationError, Exception) as e:
            logger.error("An error occured while building authorization url: %s", str(e), exc_info=True)
            raise AuthCodeGrantError("An error occured while building authorization url.") from e

        return full_authorization_url

    async def _send_request(self,
                            url: str,
                            http_method: str | HTTPMethod,
                            authentication_header: httpx.Headers | None = None,
                            headers: dict | None = None,
                            query_params: dict | None = None,
                            body_data: dict | None = None) -> httpx.Response | None:
        """
        Makes an arbitrary HTTP request.

        Args:
            url (str | httpx.URL): The base URL to which the request will be sent.
            http_method (str | HTTPMethod): The HTTP method to use for the request (e.g., "GET", "POST").
            authentication_header: httpx.Headers | None: The optional authentication HTTP header.
            headers (dict | None): Optional dictionary of HTTP headers.
            query_params (dict | None): Optional dictionary of query parameters.
            data (dict | None): Optional dictionary representing the request body.

        Returns:
            httpx.Response | None: The response from the HTTP request, or None if an error occurs.
        """
        try:
            response: httpx.Response | None = None
            merged_headers: httpx.Headers = httpx.Headers({**(headers or {}), **(authentication_header or {})})

            # Validate the incoming base url.
            self.validate_base_url(url)

            # Validate the incoming http method.
            self.validate_http_method(http_method)

            # Validate incoming header parameters.
            self.validate_headers(merged_headers)

            # Validate incoming query parameters.
            self.validate_query_parameters(query_params)

            # Validate incoming body
            self.validate_body_data(body_data)

            async with httpx.AsyncClient() as client:

                if http_method.upper() == HTTPMethod.GET.value:
                    response = await client.get(url, params=query_params, headers=merged_headers, timeout=10.0)

                if http_method.upper() == HTTPMethod.POST.value:
                    response = await client.post(url,
                                                 params=query_params,
                                                 headers=merged_headers,
                                                 json=body_data,
                                                 timeout=10.0)

                if http_method.upper() == HTTPMethod.PUT.value:
                    response = await client.put(url,
                                                params=query_params,
                                                headers=merged_headers,
                                                json=body_data,
                                                timeout=10.0)

                if http_method.upper() == HTTPMethod.DELETE.value:
                    response = await client.delete(url, params=query_params, headers=merged_headers, timeout=10.0)

                if http_method.upper() == HTTPMethod.PATCH.value:
                    response = await client.patch(url,
                                                  params=query_params,
                                                  headers=merged_headers,
                                                  json=body_data,
                                                  timeout=10.0)

                if http_method.upper() == HTTPMethod.HEAD.value:
                    response = await client.head(url, params=query_params, headers=merged_headers, timeout=10.0)

                if http_method.upper() == HTTPMethod.OPTIONS.value:
                    response = await client.options(url, params=query_params, headers=merged_headers, timeout=10.0)

        except (BaseUrlValidationError,
                HTTPMethodValidationError,
                ValidationError,
                HTTPHeaderValidationError,
                QueryParameterValidationError,
                BodyValidationError) as e:

            logger.error("An error occured while building request url: %s", str(e), exc_info=True)
            raise APIRequestError("An error occured while building request url.") from e

        except (httpx.RequestError, httpx.TimeoutException, httpx.HTTPStatusError, httpx.NetworkError) as e:
            logger.error("An error occured while sending request: %s", str(e), exc_info=True)
            raise APIRequestError("An error occured while sending request.") from e

        except Exception as e:
            logger.error("Unexpected eror occured sending request %s", str(e), exc_info=True)
            raise APIRequestError("An unexpected error occured while sending request.") from e

        return response
