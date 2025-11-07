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
"""Data models for contribution tracking and aggregation."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field


class ContributionType(str, Enum):
    """Types of contributions that can be tracked."""

    JIRA_ISSUE = "jira_issue"
    JIRA_COMMENT = "jira_comment"
    GITLAB_COMMIT = "gitlab_commit"
    GITLAB_MERGE_REQUEST = "gitlab_merge_request"
    GITLAB_REVIEW = "gitlab_review"
    CONFLUENCE_PAGE = "confluence_page"
    CONFLUENCE_COMMENT = "confluence_comment"


class ContributionItem(BaseModel):
    """A single contribution item from any platform."""

    contribution_type: ContributionType
    title: str
    description: str | None = None
    url: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        date_str = ""
        if self.created_at:
            date_str = f" ({self.created_at.strftime('%Y-%m-%d')})"
        return f"[{self.contribution_type.value}] {self.title}{date_str}"


class JiraContribution(BaseModel):
    """JIRA-specific contribution data."""

    issues_created: list[ContributionItem] = Field(default_factory=list)
    issues_updated: list[ContributionItem] = Field(default_factory=list)
    issues_resolved: list[ContributionItem] = Field(default_factory=list)
    comments_added: list[ContributionItem] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Return total number of JIRA contributions."""
        return (len(self.issues_created) + len(self.issues_updated) + len(self.issues_resolved) +
                len(self.comments_added))


class GitLabContribution(BaseModel):
    """GitLab-specific contribution data."""

    commits: list[ContributionItem] = Field(default_factory=list)
    merge_requests_created: list[ContributionItem] = Field(default_factory=list)
    merge_requests_merged: list[ContributionItem] = Field(default_factory=list)
    code_reviews: list[ContributionItem] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Return total number of GitLab contributions."""
        return (len(self.commits) + len(self.merge_requests_created) + len(self.merge_requests_merged) +
                len(self.code_reviews))


class ConfluenceContribution(BaseModel):
    """Confluence-specific contribution data."""

    pages_created: list[ContributionItem] = Field(default_factory=list)
    pages_updated: list[ContributionItem] = Field(default_factory=list)
    comments_added: list[ContributionItem] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Return total number of Confluence contributions."""
        return (len(self.pages_created) + len(self.pages_updated) + len(self.comments_added))


class AggregatedContributions(BaseModel):
    """Aggregated contributions from all platforms."""

    user_id: str
    start_date: datetime
    end_date: datetime
    jira: JiraContribution = Field(default_factory=JiraContribution)
    gitlab: GitLabContribution = Field(default_factory=GitLabContribution)
    confluence: ConfluenceContribution = Field(default_factory=ConfluenceContribution)

    @property
    def total_count(self) -> int:
        """Return total number of contributions across all platforms."""
        return self.jira.total_count + self.gitlab.total_count + self.confluence.total_count

    def summary(self) -> str:
        """Return a summary string of all contributions."""
        return (f"Contributions from {self.start_date.strftime('%Y-%m-%d')} "
                f"to {self.end_date.strftime('%Y-%m-%d')}:\n"
                f"  JIRA: {self.jira.total_count} items\n"
                f"  GitLab: {self.gitlab.total_count} items\n"
                f"  Confluence: {self.confluence.total_count} items\n"
                f"  Total: {self.total_count} items")
