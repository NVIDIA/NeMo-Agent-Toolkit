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

import httpx

from aiq.data_models.authentication import AuthResult
from aiq.data_models.authentication import HTTPResponse

logger = logging.getLogger(__name__)


class AuthProviderMixin(httpx.AsyncClient):
    """
    Mixin that extends AuthProviderBase with HTTP client capabilities.

    Provides persistent httpx.AsyncClient for connection pooling and reuse,
    while keeping HTTP configuration flexible at the request level.
    """

    def __init__(self, **kwargs):
        """
        Initialize HTTP client with default settings.

        Creates a default httpx.AsyncClient for connection pooling and reuse.
        All HTTP infrastructure configuration (timeouts, SSL, proxies, etc.)
        is handled at the request level for maximum flexibility.

        This maintains clean separation: mixin provides HTTP client capability,
        auth provider handles authentication logic, requests handle HTTP config.
        """
        # Initialize httpx.AsyncClient with sensible defaults only
        super().__init__()

    async def request(
            self,
            method: str,
            url: str,
            user_id: str | None = None,
            apply_auth: bool = True,  # Default to authenticated - most auth provider requests need auth
            headers: dict | None = None,
            params: dict | None = None,
            body_data: str | dict | None = None,
            timeout: int | None = None,
            **kwargs) -> HTTPResponse:
        """Make an HTTP request with optional authentication.

        This method provides a unified interface for both authenticated and public requests.
        Since this is a mixin for authentication providers, most requests are expected
        to be authenticated, hence the default of apply_auth=True.

        Args:
            method: HTTP method as string (GET, POST, PUT, DELETE, etc.)
            url: Target URL for the request
            user_id: User ID for authentication (uses current context if None)
            apply_auth: If True, includes authentication; if False, makes unauthenticated request
                       Default is True since most requests through auth providers need authentication.
                       Explicitly set to False for public endpoints (health checks, documentation, etc.)
            headers: Additional headers to include
            params: Query parameters
            body_data: Request body (string or dict for JSON)
            timeout: Request timeout in seconds
            **kwargs: Additional httpx request parameters

        Returns:
            HTTPResponse: Structured response with status, headers, body, and auth context

        Examples:
            # Authenticated requests (default behavior)
            user_data = await provider.request("GET", "/user/profile")
            response = await provider.request("POST", "/orders", json={"item": "widget"})

            # Public/unauthenticated requests (explicit)
            health = await provider.request("GET", "/health", apply_auth=False)
            docs = await provider.request("GET", "/api/docs", apply_auth=False)
        """

        # Validate user_id string format when provided (allow None for session fallback)
        if apply_auth and user_id is not None:
            if isinstance(user_id, str) and user_id.strip() == "":
                raise ValueError(
                    "user_id cannot be empty or whitespace-only when apply_auth=True. "
                    "Use None to trigger fallback behavior (session cookies) or provide a valid user identifier.")

        # Initialize request kwargs
        request_kwargs = {"headers": headers or {}, "params": params or {}, "timeout": timeout or 30, **kwargs}

        # Handle body data based on type
        if body_data is not None:
            if isinstance(body_data, dict):
                request_kwargs["json"] = body_data
            else:
                request_kwargs["data"] = body_data

        auth_result: AuthResult | None = None

        # Perform authentication if requested
        if apply_auth:
            try:
                # Call the authenticate method from AuthProviderBase
                # This will be available through multiple inheritance
                auth_result = await self.authenticate(user_id=user_id)  # type: ignore

                if not auth_result or not auth_result.credentials:
                    logger.warning(f"No authentication credentials received for user '{user_id}'")
                else:
                    # Automatically inject credentials into request
                    auth_result.attach(request_kwargs)

            except Exception as e:
                logger.error(f"Authentication failed for {method} {url}: {str(e)}")
                # Continue without authentication - let the API decide if auth is required

        try:
            # Make the request using inherited httpx.AsyncClient
            response = await super().request(method, url, **request_kwargs)

            # Convert httpx.Response to our structured HTTPResponse
            return self._convert_response(response, auth_result)

        except httpx.HTTPStatusError as e:
            # Handle authentication errors with automatic retry
            if e.response.status_code == 401 and apply_auth and auth_result:
                logger.info(f"Received 401 for {method} {url}, attempting token refresh")

                # Try to refresh token if the auth provider supports it
                if hasattr(self, '_attempt_token_refresh'):
                    try:
                        refreshed_auth = await self._attempt_token_refresh(user_id, auth_result)  # type: ignore
                        if refreshed_auth:
                            # Clear old auth and apply new
                            request_kwargs = {
                                k: v
                                for k, v in request_kwargs.items() if k not in ['headers', 'params', 'cookies', 'auth']
                            }
                            request_kwargs.setdefault('headers', {})
                            request_kwargs.setdefault('params', {})

                            refreshed_auth.attach(request_kwargs)

                            # Retry the request with refreshed token
                            response = await super().request(method, url, **request_kwargs)
                            return self._convert_response(response, refreshed_auth)
                    except Exception as refresh_error:
                        logger.warning(f"Token refresh failed: {refresh_error}")

            # If we can't refresh or refresh failed, convert the error response
            return self._convert_response(e.response, auth_result)

        except Exception as e:
            logger.error(f"Request failed for {method} {url}: {str(e)}")
            # Return error response in our standard format
            return HTTPResponse(status_code=500,
                                headers={},
                                body={
                                    "error": "Request failed",
                                    "message": str(e),
                                    "url": url,
                                    "method": method,
                                    "status": "failed"
                                },
                                content_type="application/json",
                                url=url,
                                auth_result=auth_result)

    def _convert_response(self, response: httpx.Response, auth_result: AuthResult | None = None) -> HTTPResponse:
        """
        Convert httpx.Response to our structured HTTPResponse format.

        This preserves all response data while adding authentication context
        and providing consistent response handling across the system.
        """
        # Parse response body intelligently
        try:
            # Try JSON first (most common for APIs)
            if response.headers.get('content-type', '').startswith('application/json'):
                body = response.json()
            else:
                # Fall back to text
                body = response.text
        except (json.JSONDecodeError, ValueError):
            # If all else fails, use raw text
            body = response.text

        return HTTPResponse(status_code=response.status_code,
                            headers=dict(response.headers),
                            body=body,
                            cookies=dict(response.cookies) if response.cookies else None,
                            content_type=response.headers.get('Content-Type'),
                            url=str(response.url),
                            elapsed=response.elapsed.total_seconds() if response.elapsed else None,
                            auth_result=auth_result)

    async def get(self, url: str, user_id: str | None = None, apply_auth: bool = True, **kwargs) -> HTTPResponse:
        """Make a GET request."""
        return await self.request("GET", url, user_id, apply_auth, **kwargs)

    async def post(self,
                   url: str,
                   user_id: str | None = None,
                   apply_auth: bool = True,
                   json: dict | None = None,
                   data: str | dict | None = None,
                   **kwargs) -> HTTPResponse:
        """Make a POST request."""
        # Handle json vs data parameters (following httpx conventions)
        body_data = json if json is not None else data
        return await self.request("POST", url, user_id, apply_auth, body_data=body_data, **kwargs)

    async def put(self,
                  url: str,
                  user_id: str | None = None,
                  apply_auth: bool = True,
                  json: dict | None = None,
                  data: str | dict | None = None,
                  **kwargs) -> HTTPResponse:
        """Make a PUT request."""
        body_data = json if json is not None else data
        return await self.request("PUT", url, user_id, apply_auth, body_data=body_data, **kwargs)

    async def delete(self, url: str, user_id: str | None = None, apply_auth: bool = True, **kwargs) -> HTTPResponse:
        """Make a DELETE request."""
        return await self.request("DELETE", url, user_id, apply_auth, **kwargs)

    async def patch(self,
                    url: str,
                    user_id: str | None = None,
                    apply_auth: bool = True,
                    json: dict | None = None,
                    data: str | dict | None = None,
                    **kwargs) -> HTTPResponse:
        """Make a PATCH request."""
        body_data = json if json is not None else data
        return await self.request("PATCH", url, user_id, apply_auth, body_data=body_data, **kwargs)

    async def head(self, url: str, user_id: str | None = None, apply_auth: bool = True, **kwargs) -> HTTPResponse:
        """Make a HEAD request."""
        return await self.request("HEAD", url, user_id, apply_auth, **kwargs)

    async def options(self, url: str, user_id: str | None = None, apply_auth: bool = True, **kwargs) -> HTTPResponse:
        """Make an OPTIONS request."""
        return await self.request("OPTIONS", url, user_id, apply_auth, **kwargs)
