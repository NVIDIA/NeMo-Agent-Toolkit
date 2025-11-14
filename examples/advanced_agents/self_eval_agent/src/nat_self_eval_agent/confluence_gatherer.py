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
"""Confluence contribution gatherer tool."""

import json
import logging
from datetime import datetime
from datetime import timedelta

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class ConfluenceGathererConfig(FunctionBaseConfig, name="confluence_gatherer"):
    """Configuration for the Confluence contribution gatherer tool."""

    default_time_period_days: int = Field(default=30,
                                          description="Default number of days to look back for contributions")


@register_function(config_type=ConfluenceGathererConfig)
async def confluence_gatherer_tool(config: ConfluenceGathererConfig, builder: Builder):  # noqa: ARG001
    """
    Register the Confluence contribution gatherer tool.

    This tool provides a high-level interface for gathering Confluence contributions.
    It returns CQL queries that can be used with Confluence MCP tools.

    In Phase 2, this will be enhanced to directly integrate with MCP clients.

    Args:
        config: Configuration for the Confluence gatherer
        builder: Builder instance

    Yields:
        The Confluence gatherer function
    """
    logger.info("Initializing Confluence Gatherer tool")

    async def _gather_confluence_contributions(user_id: str, days: int | None = None) -> str:
        """
        Generate queries for gathering Confluence contributions.

        This function returns structured information about what Confluence data to collect.
        The ReAct agent can use this information to call the appropriate Confluence MCP tools.

        Args:
            user_id: The Confluence user ID or email to gather contributions for
            days: Number of days to look back (default: from config)

        Returns:
            JSON string with Confluence query information and instructions
        """
        logger.info("Preparing Confluence contribution queries for user: %s", user_id)

        days_val = days or config.default_time_period_days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_val)

        logger.debug("Time range: %s to %s", start_date.date(), end_date.date())

        # Return query information that the agent can use
        result = {
            "platform":
                "Confluence",
            "user_id":
                user_id,
            "time_period_days":
                days_val,
            "start_date":
                start_date.isoformat(),
            "end_date":
                end_date.isoformat(),
            "queries": {
                "pages_created": {
                    "cql": (f'type=page AND creator="{user_id}" AND '
                            f'created >= {start_date.strftime("%Y-%m-%d")}'),
                    "description": "Pages created by the user"
                },
                "pages_updated": {
                    "cql": (f'type=page AND contributor="{user_id}" AND '
                            f'lastModified >= {start_date.strftime("%Y-%m-%d")}'),
                    "description": "Pages updated by the user"
                },
                "comments": {
                    "cql": (f'type=comment AND creator="{user_id}" AND '
                            f'created >= {start_date.strftime("%Y-%m-%d")}'),
                    "description": "Comments added by the user"
                }
            },
            "instructions": ("Use the Confluence MCP tools (confluence.search_content) with the provided "
                             "CQL queries to gather the user's contributions. "
                             "Aggregate the results to identify documentation and knowledge sharing contributions.")
        }

        logger.info("Confluence contribution queries prepared")
        return json.dumps(result, indent=2)

    yield _gather_confluence_contributions

    logger.info("Confluence Gatherer tool cleanup complete")
