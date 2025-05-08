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

import httpx

from aiq.authentication.authentication_manager import AuthenticationManager
from aiq.authentication.interfaces import RequestManagerBase
from aiq.authentication.response_manager import ResponseManager
from aiq.data_models.authentication import AuthenticationEndpoint
from aiq.data_models.authentication import HTTPMethod
from aiq.data_models.authentication import OAuth2Config
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class RequestManager(RequestManagerBase):

    def __init__(self, url: str, method: str, headers: dict, params: dict, data: dict) -> None:
        self._url: str = url
        self._method: str = method
        self._headers: dict = headers
        self._params: dict = params
        self._data: dict = data
        self._authentication_manager: AuthenticationManager = AuthenticationManager(self)
        self._response_manager: ResponseManager = ResponseManager()

    @property
    def authentication_manager(self) -> AuthenticationManager:
        return self._authentication_manager

    def validate_values(self, input_dict: dict) -> None:
        # Check to ensure all parameters have values
        invalid_values = [key for key, value in input_dict.items() if not value]

        if invalid_values:
            raise ValueError(f"Empty or invalid values for input: {input_dict}. "
                             f"Invalid Values: {', '.join(invalid_values)}")

    def validate_url(self, url: str) -> None:  # TODO EE: Add more verbose general url validation logic.
        if not url.startswith(("http://", "https://")):
            # TODO EE: Add custom exceptions for url
            raise ValueError("Base URL must start with http:// or https://")

    def validate_http_method(self, http_method: str) -> None:
        """
        Validates that the provided HTTP method is one of the allowed standard methods.

        Args:
            http_method (str): The HTTP method to validate (e.g., 'GET', 'POST').
        """
        try:
            HTTPMethod(http_method.upper())
        except ValueError:
            valid_http_methods = ', '.join([method.value for method in HTTPMethod])
            raise ValueError(f"Invalid HTTP method: '{http_method}'. Must be one of {valid_http_methods}.")

    def validate_headers(self, headers: dict) -> None:
        # Check to ensure all headers have values
        self.validate_values(headers)

    def validate_query_parameters(self, query_params: dict) -> None:
        # Check to ensure all query params have values
        self.validate_values(query_params)

    def validate_body_data(self, body_data) -> None:
        # TODO EE: Update
        pass

    def validate_oauth_query_params(self, query_params: dict) -> None:
        required_query_params: set = {
            "audience", "client_id", "scope", "redirect_uri", "state", "response_type", "prompt"
        }

        # Check for missing OAuth2.0 query paramters
        missing_query_paramters = [
            query_param for query_param in required_query_params if query_param not in query_params
        ]
        if missing_query_paramters:
            raise ValueError(f"Missing required query parameters: {', '.join(missing_query_paramters)}")

        # Check to ensure all OAuth2.0 query parameters have values
        self.validate_values(query_params)

    async def build_authorization_url(self,
                                      authentication_provider: str,
                                      response_type: str = "code",
                                      prompt: str = "consent") -> httpx.URL | None:
        """
        Construct an authorization URL using httpx.URL to initiate the OAuth2.0 Code Flow.

        Args:
            authentication_provider (str): The name of the registered authentication provider.

        Returns:
            httpx.URL: Constructed URL if successful, or None if error occurs.
        """
        from aiq.authentication.credentials_manager import _CredentialsManager

        try:
            authentication_provider: OAuth2Config | None = _CredentialsManager()._get_authentication_provider(
                authentication_provider)

            if authentication_provider is None:
                raise ValueError(f"Authentication provider '{authentication_provider}' not found.")

            self.validate_url(authentication_provider.authorization_url)

            query_params: dict = authentication_provider.model_dump(include={"audience", "client_id", "state"})
            query_params["scope"] = " ".join(authentication_provider.scope)
            # TODO EE: Update this so you can the exact uri including the scheme.
            query_params[
                "redirect_uri"] = f"{authentication_provider.fastapi_url}{FastApiFrontEndConfig().authorization.path}{AuthenticationEndpoint.REDIRECT_URI.value}"  # noqa: E501
            query_params["response_type"] = response_type
            query_params["prompt"] = prompt

            self.validate_oauth_query_params(query_params)

            full_authorization_url = httpx.URL(authentication_provider.authorization_url).copy_merge_params(
                query_params)  # TODO EE: NEED TO WRITE A FUNCTION TO VALIDATE FINAL URL!!!!!

        except (Exception, ValueError) as e:
            logger.error("Failed to properly construct URL for authentication provider: %s,  Error: %s",
                         authentication_provider,
                         str(e),
                         exc_info=True)
            return None

        return full_authorization_url

    async def send_authorization_request(self, authentication_provider: str):
        try:
            authorization_url: httpx.URL | None = await self.build_authorization_url(authentication_provider)

            if authorization_url is None:
                raise ValueError(f"Error occurred while building authorization URL for "
                                 f"authentication provider {authentication_provider}.")

            response: httpx.Response | None = await self.send_request(url=authorization_url,
                                                                      http_method=HTTPMethod.GET.value)

            await self._response_manager._handle_oauth_authorization_response(response, authentication_provider)

        except (ValueError) as e:
            logger.error("Failed to properly send authorization request. Error: %s", str(e), exc_info=True)

    async def send_request(self,
                           url: str | httpx.URL,
                           http_method: str,
                           headers: dict | httpx.Headers | None = None,
                           query_params: dict | httpx.QueryParams | None = None,
                           data: dict | None = None):  # TODO EE: Update return type.
        try:
            # self.validate_url(url)

            # self.validate_http_method(http_method)

            # self.validate_headers(headers)

            # self.validate_query_parameters(query_params)

            # self.validate_body_data(data)

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

        except httpx.RequestError as e:  # TODO EE: Update exceptions
            logger.error("An error occured while sending request: %s", str(e), exc_info=True)
            return None
        except (Exception, ValueError) as e:
            logger.error("Failed to validate request inputs: %s", str(e), exc_info=True)
            return None

        return response
