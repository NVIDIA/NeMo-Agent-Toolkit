# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from pydantic import ValidationError


@pytest.mark.parametrize("issue_number", [103, "103"])
def test_update_issue_accepts_positive_integer(issue_number: int | str):
    from nat.tool.github_tools import GithubUpdateIssueModel

    issue = GithubUpdateIssueModel(issue_number=issue_number)

    assert issue.issue_number == 103


@pytest.mark.parametrize("issue_number", [0, -1, "0", "../../other-endpoint"])
def test_update_issue_rejects_non_positive_or_non_integer_issue_number(issue_number: int | str):
    from nat.tool.github_tools import GithubUpdateIssueModel

    with pytest.raises(ValidationError):
        GithubUpdateIssueModel(issue_number=issue_number)


def test_parse_repo_name_allows_periods_in_components():
    from nat.tool.github_tools import _parse_repo_name

    assert _parse_repo_name("owner.name/repo.name") == ("owner.name", "repo.name")


@pytest.mark.parametrize("repo_name", ["owner", "owner/repo/extra", "../repo", "owner/..", "owner/repo?x=1"])
def test_parse_repo_name_rejects_path_syntax(repo_name: str):
    from nat.tool.github_tools import _parse_repo_name

    with pytest.raises(ValueError):
        _parse_repo_name(repo_name)


async def test_branch_validation_allows_slashes_and_periods():
    from nat.tool.github_tools import _validate_and_encode_branch

    assert await _validate_and_encode_branch("feature/release.1") == "feature/release.1"


async def test_branch_validation_encodes_valid_path_component_characters():
    from nat.tool.github_tools import _validate_and_encode_branch

    assert await _validate_and_encode_branch("feature/100%") == "feature/100%25"


@pytest.mark.parametrize("branch", [".", "feature/../other", "a..b"])
async def test_branch_path_syntax_is_rejected_before_git_validation(branch: str, monkeypatch: pytest.MonkeyPatch):
    from nat.tool import github_tools

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("git validation must not run for invalid path syntax")

    monkeypatch.setattr(github_tools.asyncio, "create_subprocess_exec", fail_if_called)

    with pytest.raises(ValueError):
        await github_tools._validate_and_encode_branch(branch)
