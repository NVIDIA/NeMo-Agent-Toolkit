# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.authentication.interfaces import AuthenticationClientBase
from aiq.authentication.interfaces import OAuthClientBase
from aiq.data_models.authentication import AuthenticatedContext
from aiq.data_models.authentication import CredentialLocation
from aiq.data_models.authentication import HeaderAuthScheme

logger = logging.getLogger(__name__)


class AuthenticationClientTester:
    """
    Comprehensive authentication client testing class.

    Tests custom authentication client implementations against the AuthenticationClientBase interface.
    This class validates that authentication clients properly implement the required methods for
    credential validation and authentication context construction.

    This tester is designed for non-OAuth authentication clients. For OAuth clients, use OAuth2FlowTester instead.
    """

    def __init__(self, auth_client: AuthenticationClientBase):
        """
        Initialize the authentication client tester.

        Args:
            auth_client (AuthenticationClientBase): The authentication client to test.
                Must be an instance of AuthenticationClientBase but NOT an instance of OAuthClientBase.

        Raises:
            AssertionError: If the auth_client is not an instance of AuthenticationClientBase
                           or if it's an instance of OAuthClientBase.
        """
        # Validate the client is an instance of AuthenticationClientBase
        assert isinstance(auth_client, AuthenticationClientBase), \
            "Authentication client must be an instance of AuthenticationClientBase"

        # Explicitly reject OAuthClientBase instances with clear message
        if isinstance(auth_client, OAuthClientBase):
            raise AssertionError("OAuth clients should use OAuth2FlowTester instead. "
                                 "This tester is designed for non-OAuth authentication clients only. "
                                 "Please use OAuth2FlowTester for testing OAuth authentication clients.")

        self._auth_client = auth_client

    async def run(self) -> bool:
        """
        Main entry point for running the complete authentication client test suite.

        Handles all setup, execution, and validation:
        - Tests credential validation functionality
        - Tests authentication context construction across different credential locations and schemes
        - Validates authentication context structure and content
        - Returns False immediately on any assertion failure

        Returns:
            bool: True if all tests pass successfully, False if any assertion fails
        """
        try:
            # Test credential validation
            await self._test_credential_validation()

            # Test authentication context construction
            await self._test_authentication_context_construction()

            logger.info("All authentication client tests passed successfully")
            return True
        except AssertionError as e:
            logger.error("Authentication client test failed: %s", str(e))
            return False

    async def _test_credential_validation(self) -> None:
        """Test credential validation functionality."""
        # Test validate_credentials method
        is_valid = await self._auth_client.validate_credentials()
        assert isinstance(is_valid, bool), \
            f"validate_credentials must return bool, got {type(is_valid)}"

        # Assert that credentials are valid - this ensures the test fails if credentials are invalid
        assert is_valid, \
            f"Credentials validation failed for client '{self._auth_client.config_name}'. " \
            f"Please ensure the authentication configuration is valid."

        # Log the validation result for debugging
        logger.info("Credential validation result: %s", is_valid)

    async def _test_authentication_context_construction(self) -> None:
        """Test authentication context construction across different scenarios."""
        # Test all combinations of credential locations and header schemes
        credential_locations = [
            CredentialLocation.HEADER, CredentialLocation.QUERY, CredentialLocation.COOKIE, CredentialLocation.BODY
        ]

        header_schemes = [
            HeaderAuthScheme.BEARER, HeaderAuthScheme.X_API_KEY, HeaderAuthScheme.BASIC, HeaderAuthScheme.CUSTOM
        ]

        for credential_location in credential_locations:
            for header_scheme in header_schemes:
                await self._test_authentication_context_for_location_and_scheme(credential_location, header_scheme)

    async def _test_authentication_context_for_location_and_scheme(self,
                                                                   credential_location: CredentialLocation,
                                                                   header_scheme: HeaderAuthScheme) -> None:
        """
        Test authentication context construction for a specific credential location and header scheme.

        Args:
            credential_location (CredentialLocation): The credential location to test
            header_scheme (HeaderAuthScheme): The header scheme to test
        """
        # Call construct_authentication_context
        auth_context = await self._auth_client.construct_authentication_context(credential_location=credential_location,
                                                                                header_scheme=header_scheme)

        # Validate the return type
        assert auth_context is None or isinstance(auth_context, AuthenticatedContext), \
            (f"construct_authentication_context must return AuthenticatedContext or None, "
             f"got {type(auth_context)} for location={credential_location.value}, "
             f"scheme={header_scheme.value}")

        # If context is returned, validate its structure
        if auth_context is not None:
            await self._validate_authentication_context(auth_context, credential_location, header_scheme)

        logger.info("Authentication context test passed for location=%s, scheme=%s, result=%s",
                    credential_location.value,
                    header_scheme.value,
                    "Success" if auth_context else "None")

    async def _validate_authentication_context(self,
                                               auth_context: AuthenticatedContext,
                                               credential_location: CredentialLocation,
                                               header_scheme: HeaderAuthScheme) -> None:
        """
        Validate the structure and content of an authentication context.

        Args:
            auth_context (AuthenticatedContext): The authentication context to validate
            credential_location (CredentialLocation): The credential location used
            header_scheme (HeaderAuthScheme): The header scheme used
        """
        # Validate AuthenticatedContext structure
        assert hasattr(auth_context, 'headers'), "AuthenticatedContext must have 'headers' attribute"
        assert hasattr(auth_context, 'query_params'), "AuthenticatedContext must have 'query_params' attribute"
        assert hasattr(auth_context, 'cookies'), "AuthenticatedContext must have 'cookies' attribute"
        assert hasattr(auth_context, 'body'), "AuthenticatedContext must have 'body' attribute"

        logger.info("Validating authentication context structure for location=%s, scheme=%s",
                    credential_location.value,
                    header_scheme.value)

        # Validate content based on credential location
        # Note: Some attribute values may be None if the combination is not supported
        if credential_location == CredentialLocation.HEADER:
            self._validate_header_authentication_context(auth_context, header_scheme)
        elif credential_location == CredentialLocation.QUERY:
            self._validate_query_authentication_context(auth_context)
        elif credential_location == CredentialLocation.COOKIE:
            self._validate_cookie_authentication_context(auth_context)
        elif credential_location == CredentialLocation.BODY:
            self._validate_body_authentication_context(auth_context)

    def _validate_header_authentication_context(self,
                                                auth_context: AuthenticatedContext,
                                                header_scheme: HeaderAuthScheme) -> None:
        """
        Validate header-based authentication context.

        Args:
            auth_context (AuthenticatedContext): The authentication context to validate
            header_scheme (HeaderAuthScheme): The header scheme used
        """
        # Headers may be None if the scheme is not supported - this is acceptable
        if auth_context.headers is None:
            logger.info("Headers are None for scheme %s - this may indicate unsupported scheme", header_scheme.value)
            return

        # Validate headers type
        assert isinstance(auth_context.headers, (dict, httpx.Headers)), \
            f"Headers must be dict or httpx.Headers, got {type(auth_context.headers)}"

        # Convert to dict for easier validation
        headers_dict = dict(auth_context.headers) if isinstance(auth_context.headers,
                                                                httpx.Headers) else auth_context.headers

        # Validate headers are not empty when provided
        assert len(headers_dict) > 0, "Headers dictionary must not be empty when provided"

        # Validate header values are strings
        for key, value in headers_dict.items():
            assert isinstance(key, str), f"Header key must be string, got {type(key)}"
            assert isinstance(value, str), f"Header value must be string, got {type(value)}"

        # Log header information for debugging
        logger.info("Header authentication context validated: scheme=%s, headers=%s", header_scheme.value, headers_dict)

    def _validate_query_authentication_context(self, auth_context: AuthenticatedContext) -> None:
        """
        Validate query parameter-based authentication context.

        Args:
            auth_context (AuthenticatedContext): The authentication context to validate
        """
        if auth_context.query_params is not None:
            assert isinstance(auth_context.query_params, (dict, httpx.QueryParams)), \
                f"Query params must be dict or httpx.QueryParams, got {type(auth_context.query_params)}"

            # Convert to dict for easier validation
            query_dict = dict(auth_context.query_params) if isinstance(auth_context.query_params,
                                                                       httpx.QueryParams) else auth_context.query_params

            # Validate query parameters are not empty when provided
            assert len(query_dict) > 0, "Query parameters must not be empty when provided"

            # Validate query parameter values are strings
            for key, value in query_dict.items():
                assert isinstance(key, str), f"Query param key must be string, got {type(key)}"
                assert isinstance(value, str), f"Query param value must be string, got {type(value)}"

            logger.info("Query authentication context validated: params=%s", query_dict)
        else:
            logger.info("Query params are None - this may indicate unsupported credential location")

    def _validate_cookie_authentication_context(self, auth_context: AuthenticatedContext) -> None:
        """
        Validate cookie-based authentication context.

        Args:
            auth_context (AuthenticatedContext): The authentication context to validate
        """
        if auth_context.cookies is not None:
            assert isinstance(auth_context.cookies, (dict, httpx.Cookies)), \
                f"Cookies must be dict or httpx.Cookies, got {type(auth_context.cookies)}"

            # Convert to dict for easier validation
            cookies_dict = dict(auth_context.cookies) if isinstance(auth_context.cookies,
                                                                    httpx.Cookies) else auth_context.cookies

            # Validate cookies are not empty when provided
            assert len(cookies_dict) > 0, "Cookies must not be empty when provided"

            # Validate cookie values are strings
            for key, value in cookies_dict.items():
                assert isinstance(key, str), f"Cookie key must be string, got {type(key)}"
                assert isinstance(value, str), f"Cookie value must be string, got {type(value)}"

            logger.info("Cookie authentication context validated: cookies=%s", cookies_dict)
        else:
            logger.info("Cookies are None - this may indicate unsupported credential location")

    def _validate_body_authentication_context(self, auth_context: AuthenticatedContext) -> None:
        """
        Validate body-based authentication context.

        Args:
            auth_context (AuthenticatedContext): The authentication context to validate
        """
        if auth_context.body is not None:
            assert isinstance(auth_context.body, dict), \
                f"Body must be dict, got {type(auth_context.body)}"

            # Validate body is not empty when provided
            assert len(auth_context.body) > 0, "Body must not be empty when provided"

            # Validate body values are strings
            for key, value in auth_context.body.items():
                assert isinstance(key, str), f"Body key must be string, got {type(key)}"
                assert isinstance(value, str), f"Body value must be string, got {type(value)}"

            logger.info("Body authentication context validated: body=%s", auth_context.body)
        else:
            logger.info("Body is None - this may indicate unsupported credential location")

    async def test_validate_credentials(self) -> bool:
        """
        Test the validate_credentials method specifically.

        Returns:
            bool: The result of validate_credentials()
        """
        is_valid = await self._auth_client.validate_credentials()
        assert isinstance(is_valid, bool), \
            f"validate_credentials must return bool, got {type(is_valid)}"
        return is_valid

    async def test_construct_authentication_context(self,
                                                    credential_location: CredentialLocation,
                                                    header_scheme: HeaderAuthScheme) -> AuthenticatedContext | None:
        """
        Test the construct_authentication_context method specifically.

        Args:
            credential_location (CredentialLocation): The credential location to test
            header_scheme (HeaderAuthScheme): The header scheme to test

        Returns:
            AuthenticatedContext | None: The result of construct_authentication_context()
        """
        auth_context = await self._auth_client.construct_authentication_context(credential_location=credential_location,
                                                                                header_scheme=header_scheme)

        assert auth_context is None or isinstance(auth_context, AuthenticatedContext), \
            (f"construct_authentication_context must return AuthenticatedContext or None, "
             f"got {type(auth_context)}")

        if auth_context is not None:
            await self._validate_authentication_context(auth_context, credential_location, header_scheme)

        return auth_context
