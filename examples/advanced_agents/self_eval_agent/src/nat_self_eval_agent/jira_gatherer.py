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
"""JIRA contribution gatherer tool."""

import json
import logging
from datetime import datetime
from datetime import timedelta

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class JiraGathererConfig(FunctionBaseConfig, name="jira_gatherer"):
    """Configuration for the JIRA contribution gatherer tool."""

    default_time_period_days: int = Field(default=30,
                                          description="Default number of days to look back for contributions")


@register_function(config_type=JiraGathererConfig)
async def jira_gatherer_tool(config: JiraGathererConfig, builder: Builder):  # noqa: ARG001
    """
    Register the JIRA contribution gatherer tool.

    This tool provides a high-level interface for gathering JIRA contributions.
    It returns JQL queries and parameters that can be used with JIRA MCP tools.

    In Phase 2, this will be enhanced to directly integrate with MCP clients.

    Args:
        config: Configuration for the JIRA gatherer
        builder: Builder instance

    Yields:
        The JIRA gatherer function
    """
    logger.info("Initializing JIRA Gatherer tool")

    async def _gather_jira_contributions(user_id: str, days: int | None = None) -> str:
        """
        Generate queries for gathering JIRA contributions.

        This function returns structured information about what JIRA data to collect.
        The ReAct agent can use this information to call the appropriate JIRA MCP tools.

        Args:
            user_id: The JIRA user ID or email to gather contributions for
            days: Number of days to look back (default: from config)

        Returns:
            JSON string with JIRA query information and instructions
        """
        logger.info("Preparing JIRA contribution queries for user: %s", user_id)

        days_val = days or config.default_time_period_days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_val)

        logger.debug("Time range: %s to %s", start_date.date(), end_date.date())

        # Return query information that the agent can use
        result = {
            "platform":
                "JIRA",
            "user_id":
                user_id,
            "time_period_days":
                days_val,
            "start_date":
                start_date.isoformat(),
            "end_date":
                end_date.isoformat(),
            "queries": {
                "issues_created": {
                    "jql": f'creator = "{user_id}" AND created >= -{days_val}d',
                    "description": "Issues created by the user"
                },
                "issues_updated": {
                    "jql": f'assignee = "{user_id}" AND updated >= -{days_val}d',
                    "description": "Issues assigned to and updated by the user"
                },
                "issues_resolved": {
                    "jql": (f'assignee = "{user_id}" AND '
                            f'status in (Resolved, Closed, Done) AND '
                            f'resolved >= -{days_val}d'),
                    "description": "Issues resolved or closed by the user"
                }
            },
            "instructions": ("Use the JIRA MCP tools (jira.search_issues) with the provided "
                             "JQL queries to gather the user's contributions. "
                             "Aggregate the results to identify major contributions.")
        }

        logger.info("JIRA contribution queries prepared")
        return json.dumps(result, indent=2)

    yield _gather_jira_contributions

    logger.info("JIRA Gatherer tool cleanup complete")
