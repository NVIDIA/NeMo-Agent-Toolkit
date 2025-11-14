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
import importlib
import logging
import typing

from pydantic import SecretStr

from nat.authentication.interfaces import AuthProviderBase
from nat.data_models.authentication import AuthResult
from nat.data_models.authentication import Credential
from nat.data_models.authentication import HeaderCred
from nat.plugins.mcp.auth.service_account.provider_config import MCPServiceAccountProviderConfig
from nat.plugins.mcp.auth.service_account.token_client import ServiceAccountTokenClient

logger = logging.getLogger(__name__)


class MCPServiceAccountProvider(AuthProviderBase[MCPServiceAccountProviderConfig]):
    """
    MCP service account authentication provider using OAuth2 client credentials.

    Provides headless authentication for MCP clients using service account credentials.
    Supports two authentication patterns:

    1. Single authentication: OAuth2 service account token only
    2. Dual authentication: OAuth2 service account token + service-specific token

    """

    def __init__(self, config: MCPServiceAccountProviderConfig, builder=None):
        super().__init__(config)

        # Initialize token client
        self._token_client = ServiceAccountTokenClient(
            client_id=config.client_id,
            client_secret=config.client_secret,
            token_url=config.token_url,
            scopes=" ".join(config.scopes),  # Convert list to space-delimited string for OAuth2
            token_cache_buffer_seconds=config.token_cache_buffer_seconds,
        )

        # Load dynamic service token function if configured
        self._service_token_function = None
        if config.service_token and config.service_token.function:
            self._service_token_function = self._load_function(config.service_token.function)

        logger.info("Initialized MCP service account auth provider: "
                    "token_url=%s, scopes=%s, has_service_token=%s",
                    config.token_url,
                    config.scopes,
                    config.service_token is not None)

    def _load_function(self, function_path: str) -> typing.Callable:
        """Load a Python function from a module path string (e.g., 'my_module.get_token')."""
        try:
            module_name, func_name = function_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
            logger.info("Loaded service token function: %s", function_path)
            return func
        except Exception as e:
            raise ValueError(f"Failed to load service token function '{function_path}': {e}") from e

    async def authenticate(self, user_id: str | None = None, **kwargs) -> AuthResult:
        """
        Authenticate using OAuth2 client credentials flow.

        Note: user_id is ignored for service accounts (non-session-specific).

        Returns:
            AuthResult with HeaderCred objects for service account authentication
        """
        # Get OAuth2 access token (cached if still valid)
        access_token = await self._token_client.get_access_token()

        # Build credentials list using HeaderCred
        credentials: list[Credential] = [
            HeaderCred(name="Authorization", value=SecretStr(f"Bearer {access_token.get_secret_value()}"))
        ]

        # Add service-specific token if configured
        if self.config.service_token:
            service_header = self.config.service_token.header
            service_token_value = None

            # Get service token from static config or dynamic function
            if self.config.service_token.token:
                # Static token from config
                service_token_value = self.config.service_token.token.get_secret_value()

            elif self._service_token_function:
                # Dynamic token from function
                try:
                    # Pass configured kwargs to the function
                    # Function can access runtime context via AIQContext.get() if needed
                    # Handle both sync and async functions
                    if asyncio.iscoroutinefunction(self._service_token_function):
                        result = await self._service_token_function(**self.config.service_token.kwargs)
                    else:
                        result = self._service_token_function(**self.config.service_token.kwargs)

                    # Handle function return type: str or tuple[str, str]
                    if isinstance(result, tuple):
                        service_header, service_token_value = result
                    else:
                        service_token_value = result

                    logger.debug("Retrieved service token via dynamic function")

                except Exception as e:
                    raise RuntimeError(f"Failed to get service token from function: {e}") from e

            if service_token_value:
                credentials.append(HeaderCred(name=service_header, value=SecretStr(service_token_value)))

        # Return AuthResult with HeaderCred objects
        return AuthResult(
            credentials=credentials,
            token_expires_at=self._token_client.token_expires_at,
            raw={},
        )
