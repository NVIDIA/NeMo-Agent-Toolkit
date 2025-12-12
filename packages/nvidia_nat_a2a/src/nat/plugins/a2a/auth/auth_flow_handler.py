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

from nat.data_models.authentication import AuthenticatedContext
from nat.data_models.authentication import AuthFlowType
from nat.data_models.authentication import AuthProviderBaseConfig
from nat.front_ends.console.authentication_flow_handler import ConsoleAuthenticationFlowHandler

logger = logging.getLogger(__name__)


class A2AAuthenticationFlowHandler(ConsoleAuthenticationFlowHandler):
    """
    Authentication helper for A2A client environments.

    This handler is specifically designed for A2A client scenarios where
    authentication needs to happen before the default auth_callback is available
    in the Context. It handles OAuth2 authorization code flow during A2A client
    connection and tool discovery phases.

    Key differences from console handler:
    - Optimized for A2A client connection workflows
    - Designed for authentication during workflow building
    - Handles authentication before session context is established

    Usage:
        This handler is typically used in CLI environments where A2A clients
        need to authenticate during workflow initialization, before the standard
        session-based authentication callback is available.
    """

    def __init__(self):
        """Initialize the A2A authentication flow handler."""
        super().__init__()
        logger.debug("A2AAuthenticationFlowHandler initialized")

    async def authenticate(self, config: AuthProviderBaseConfig, method: AuthFlowType) -> AuthenticatedContext:
        """
        Handle authentication for A2A client environments.

        This method delegates to the parent ConsoleAuthenticationFlowHandler but
        provides A2A-specific logging and can be extended with A2A-specific logic
        if needed in the future.

        Args:
            config: Authentication provider configuration
            method: Authentication flow type (OAuth2, HTTP Basic, etc.)

        Returns:
            AuthenticatedContext with credentials for A2A server access

        Raises:
            ValueError: If config is invalid
            NotImplementedError: If method is not supported
            RuntimeError: If authentication fails
        """
        logger.info("Starting A2A client authentication flow (method: %s)", method)

        try:
            # Delegate to parent ConsoleAuthenticationFlowHandler
            result = await super().authenticate(config, method)
            logger.info("A2A client authentication successful")
            return result
        except Exception as e:
            logger.error("A2A client authentication failed: %s", e)
            raise
