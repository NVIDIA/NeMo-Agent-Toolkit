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
"""GitLab contribution gatherer tool."""

import json
import logging
from datetime import datetime
from datetime import timedelta

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class GitLabGathererConfig(FunctionBaseConfig, name="gitlab_gatherer"):
    """Configuration for the GitLab contribution gatherer tool."""

    default_time_period_days: int = Field(default=30,
                                          description="Default number of days to look back for contributions")


@register_function(config_type=GitLabGathererConfig)
async def gitlab_gatherer_tool(config: GitLabGathererConfig, builder: Builder):  # noqa: ARG001
    """
    Register the GitLab contribution gatherer tool.

    This tool provides a high-level interface for gathering GitLab contributions.
    It returns query parameters that can be used with GitLab MCP tools.

    In Phase 2, this will be enhanced to directly integrate with MCP clients.

    Args:
        config: Configuration for the GitLab gatherer
        builder: Builder instance

    Yields:
        The GitLab gatherer function
    """
    logger.info("Initializing GitLab Gatherer tool")

    async def _gather_gitlab_contributions(user_id: str, days: int | None = None) -> str:
        """
        Generate queries for gathering GitLab contributions.

        This function returns structured information about what GitLab data to collect.
        The ReAct agent can use this information to call the appropriate GitLab MCP tools.

        Args:
            user_id: The GitLab user ID or username to gather contributions for
            days: Number of days to look back (default: from config)

        Returns:
            JSON string with GitLab query information and instructions
        """
        logger.info("Preparing GitLab contribution queries for user: %s", user_id)

        days_val = days or config.default_time_period_days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_val)

        logger.debug("Time range: %s to %s", start_date.date(), end_date.date())

        # Return query information that the agent can use
        result = {
            "platform":
                "GitLab",
            "user_id":
                user_id,
            "time_period_days":
                days_val,
            "start_date":
                start_date.isoformat(),
            "end_date":
                end_date.isoformat(),
            "queries": {
                "commits": {
                    "tool": "gitlab.get_user_commits",
                    "params": {
                        "user_id": user_id, "since": start_date.isoformat(), "until": end_date.isoformat()
                    },
                    "description": "Commits authored by the user"
                },
                "merge_requests": {
                    "tool": "gitlab.get_merge_requests",
                    "params": {
                        "author_id": user_id,
                        "created_after": start_date.isoformat(),
                        "updated_after": start_date.isoformat()
                    },
                    "description": "Merge requests created by the user"
                },
                "code_reviews": {
                    "tool": "gitlab.get_merge_request_reviews",
                    "params": {
                        "reviewer_id": user_id, "updated_after": start_date.isoformat()
                    },
                    "description": "Code reviews provided by the user"
                }
            },
            "instructions": ("Use the GitLab MCP tools with the provided parameters to gather "
                             "the user's contributions. Analyze commits, merge requests, and reviews "
                             "to identify major contributions.")
        }

        logger.info("GitLab contribution queries prepared")
        return json.dumps(result, indent=2)

    yield _gather_gitlab_contributions

    logger.info("GitLab Gatherer tool cleanup complete")
