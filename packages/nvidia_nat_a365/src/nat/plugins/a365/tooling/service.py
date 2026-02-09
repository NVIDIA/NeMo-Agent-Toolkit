# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""A365 tooling service for MCP server discovery and configuration."""

import logging

from microsoft_agents_a365.tooling import MCPServerConfig, McpToolServerConfigurationService


class A365ToolingService:
    """Service for discovering and configuring MCP tool servers from Agent 365.

    This service wraps the A365 SDK's McpToolServerConfigurationService to provide
    a NAT-friendly interface for discovering MCP servers configured for an agent.
    """

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize the A365 tooling service.

        Args:
            logger: Optional logger instance. Defaults to module logger if not provided.
        """
        self._logger = logger or logging.getLogger(__name__)
        self._service = McpToolServerConfigurationService(logger=self._logger)

    async def list_tool_servers(
        self,
        agentic_app_id: str,
        auth_token: str,
    ) -> list[MCPServerConfig]:
        """Get the list of MCP servers configured for the agent.

        In development mode (ENVIRONMENT=Development), reads from ToolingManifest.json.
        In production mode, retrieves configuration from the A365 tooling gateway.

        Args:
            agentic_app_id: The Agent 365 agentic app ID.
            auth_token: Authentication token for accessing the tooling gateway.

        Returns:
            List of MCPServerConfig objects representing configured MCP servers.

        Raises:
            ValueError: If required parameters are invalid or empty.
            Exception: If there's an error communicating with the tooling gateway
                or reading the manifest file.

        Example:
            ```python
            servers = await service.list_tool_servers(
                agentic_app_id="my-agent-123",
                auth_token="bearer-token-here"
            )
            ```
        """
        self._logger.info(f"Listing MCP tool servers for agent {agentic_app_id}")
        servers = await self._service.list_tool_servers(
            agentic_app_id=agentic_app_id,
            auth_token=auth_token,
        )

        self._logger.info(f"Found {len(servers)} MCP tool servers")
        return servers
