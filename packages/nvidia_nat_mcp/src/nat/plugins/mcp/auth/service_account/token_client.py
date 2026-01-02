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

import asyncio
import base64
import logging
from datetime import datetime
from datetime import timedelta

import httpx
from pydantic import SecretStr

logger = logging.getLogger(__name__)


class ServiceAccountTokenClient:
    """
    Generic OAuth2 client credentials token client for service accounts.

    Implements standard OAuth2 client credentials flow with token caching.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: SecretStr,
        token_url: str,
        scopes: str,
        token_cache_buffer_seconds: int = 300,
    ):
        """
        Initialize service account token client.

        Args:
            client_id: OAuth2 client identifier
            client_secret: OAuth2 client secret (SecretStr)
            token_url: OAuth2 token endpoint URL
            scopes: Space-separated list of scopes
            token_cache_buffer_seconds: Seconds before expiry to refresh (default: 5 min)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.scopes = scopes
        self.token_cache_buffer_seconds = token_cache_buffer_seconds

        # Token cache
        self._cached_token: SecretStr | None = None
        self._token_expires_at: datetime | None = None
        self._lock = None  # Will be initialized as asyncio.Lock when needed

    @property
    def token_expires_at(self) -> datetime | None:
        return self._token_expires_at

    async def _get_lock(self) -> asyncio.Lock:
        """Lazy initialization of asyncio.Lock."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _is_token_valid(self) -> bool:
        """Check if cached token is still valid (with buffer time)."""
        if not self._cached_token or not self._token_expires_at:
            return False
        buffer = timedelta(seconds=self.token_cache_buffer_seconds)
        return datetime.now() < (self._token_expires_at - buffer)

    async def get_access_token(self) -> SecretStr:
        """
        Get OAuth2 access token, using cache if valid.

        Returns:
            Access token as SecretStr

        Raises:
            RuntimeError: If token acquisition fails
        """
        # Fast path: check cache without lock
        if self._is_token_valid():
            logger.debug("Using cached service account token")
            assert self._cached_token is not None  # _is_token_valid() ensures this
            return self._cached_token

        # Slow path: acquire lock and refresh token
        lock = await self._get_lock()
        async with lock:
            # Double-check after acquiring lock
            if self._is_token_valid():
                logger.debug("Using cached service account token (acquired during lock wait)")
                assert self._cached_token is not None  # _is_token_valid() ensures this
                return self._cached_token

            logger.info("Fetching new service account token")
            return await self._fetch_new_token()

    async def _fetch_new_token(self) -> SecretStr:
        """
        Fetch a new token from the OAuth2 token endpoint.

        Returns:
            New access token as SecretStr

        Raises:
            RuntimeError: If token request fails
        """
        # Encode credentials for Basic authentication
        credentials = f"{self.client_id}:{self.client_secret.get_secret_value()}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        headers = {"Authorization": f"Basic {encoded_credentials}", "Content-Type": "application/x-www-form-urlencoded"}

        data = {"grant_type": "client_credentials", "scope": self.scopes}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.token_url, headers=headers, data=data)

                if response.status_code == 200:
                    token_data = response.json()

                    # Cache the token
                    access_token = token_data.get("access_token")
                    if not access_token:
                        raise RuntimeError("Access token not found in token response")
                    self._cached_token = SecretStr(access_token)
                    expires_in = token_data.get("expires_in", 3600)
                    self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)

                    logger.info("Service account token acquired (expires in %ss)", expires_in)
                    return self._cached_token

                elif response.status_code == 401:
                    raise RuntimeError("Invalid service account credentials")
                elif response.status_code == 429:
                    raise RuntimeError("Service account rate limit exceeded")
                else:
                    raise RuntimeError(
                        f"Service account token request failed: {response.status_code} - {response.text}")

        except httpx.TimeoutException as e:
            raise RuntimeError(f"Service account token request timed out: {e}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Service account token request failed: {e}") from e
