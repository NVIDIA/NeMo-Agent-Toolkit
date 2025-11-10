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

"""Tests for data models."""

from datetime import datetime

import pytest

from nat_self_eval_agent.models import AggregatedContributions
from nat_self_eval_agent.models import ConfluenceContribution
from nat_self_eval_agent.models import ContributionItem
from nat_self_eval_agent.models import ContributionType
from nat_self_eval_agent.models import GitLabContribution
from nat_self_eval_agent.models import JiraContribution


def test_contribution_item_creation():
    """Test creating a contribution item."""
    item = ContributionItem(
        contribution_type=ContributionType.JIRA_ISSUE,
        title="TEST-123",
        description="Test issue",
        url="https://jira.example.com/browse/TEST-123",
        created_at=datetime(2025, 1, 1),
        metadata={"status": "Done"}
    )

    assert item.contribution_type == ContributionType.JIRA_ISSUE
    assert item.title == "TEST-123"
    assert item.metadata["status"] == "Done"
    assert "TEST-123" in str(item)
    assert "2025-01-01" in str(item)


def test_jira_contribution_totals():
    """Test JIRA contribution counting."""
    contribution = JiraContribution()

    # Add some items
    contribution.issues_created.append(
        ContributionItem(
            contribution_type=ContributionType.JIRA_ISSUE,
            title="TEST-1",
            description="Issue 1"
        )
    )
    contribution.issues_resolved.append(
        ContributionItem(
            contribution_type=ContributionType.JIRA_ISSUE,
            title="TEST-2",
            description="Issue 2"
        )
    )

    assert contribution.total_count == 2
    assert len(contribution.issues_created) == 1
    assert len(contribution.issues_resolved) == 1


def test_gitlab_contribution_totals():
    """Test GitLab contribution counting."""
    contribution = GitLabContribution()

    # Add items
    contribution.commits.append(
        ContributionItem(
            contribution_type=ContributionType.GITLAB_COMMIT,
            title="feat: add feature",
            description="Commit message"
        )
    )
    contribution.merge_requests_created.append(
        ContributionItem(
            contribution_type=ContributionType.GITLAB_MERGE_REQUEST,
            title="MR !123",
            description="Merge request"
        )
    )

    assert contribution.total_count == 2


def test_confluence_contribution_totals():
    """Test Confluence contribution counting."""
    contribution = ConfluenceContribution()

    # Add items
    contribution.pages_created.append(
        ContributionItem(
            contribution_type=ContributionType.CONFLUENCE_PAGE,
            title="Getting Started",
            description="Documentation page"
        )
    )

    assert contribution.total_count == 1


def test_aggregated_contributions():
    """Test aggregated contributions from all platforms."""
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31)

    aggregated = AggregatedContributions(
        user_id="john.doe@example.com",
        start_date=start_date,
        end_date=end_date
    )

    # Add JIRA contributions
    aggregated.jira.issues_created.append(
        ContributionItem(
            contribution_type=ContributionType.JIRA_ISSUE,
            title="TEST-1",
            description="Issue 1"
        )
    )

    # Add GitLab contributions
    aggregated.gitlab.commits.append(
        ContributionItem(
            contribution_type=ContributionType.GITLAB_COMMIT,
            title="feat: add feature",
            description="Commit"
        )
    )

    # Add Confluence contributions
    aggregated.confluence.pages_created.append(
        ContributionItem(
            contribution_type=ContributionType.CONFLUENCE_PAGE,
            title="Documentation",
            description="Page"
        )
    )

    # Test total count
    assert aggregated.total_count == 3
    assert aggregated.jira.total_count == 1
    assert aggregated.gitlab.total_count == 1
    assert aggregated.confluence.total_count == 1

    # Test summary
    summary = aggregated.summary()
    assert "2025-01-01" in summary
    assert "2025-01-31" in summary
    assert "JIRA: 1" in summary
    assert "GitLab: 1" in summary
    assert "Confluence: 1" in summary
    assert "Total: 3" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
