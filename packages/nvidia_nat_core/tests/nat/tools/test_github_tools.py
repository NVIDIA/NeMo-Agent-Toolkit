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

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
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


async def test_get_issue_sends_labels_as_one_comma_separated_value(monkeypatch: pytest.MonkeyPatch):
    from nat.tool.github_tools import GithubGetIssueModel
    from nat.tool.github_tools import GithubGetIssueModelList
    from nat.tool.github_tools import GithubGroupConfig
    from nat.tool.github_tools import github_tool

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    real_async_client = httpx.AsyncClient

    def async_client_with_mock_transport(*args, **kwargs):
        return real_async_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(httpx, "AsyncClient", async_client_with_mock_transport)

    async with github_tool(GithubGroupConfig(repo_name="owner/repo"), MagicMock()) as group:
        get_issue = (await group.get_all_functions())["github__get_issue"]
        result = await get_issue.ainvoke(
            GithubGetIssueModelList(filter_parameters=[GithubGetIssueModel(labels=["bug", "Needs Triage"])]))

    assert json.loads(result) == [[]]
    assert len(requests) == 1
    # The GitHub API reads `labels` as one comma-separated value; repeating the parameter keeps only the last label.
    assert requests[0].url.params.get_list("labels") == ["bug,Needs Triage"]


@pytest.mark.parametrize(
    "ref",
    [
        "2545d2c9d34a948cf776471a305742986268dc92",  # commit permalink
        "v1.9.0",  # tag
        "develop",  # branch
        "feature/new-parser",  # branch with a slash
    ])
async def test_get_file_requests_the_raw_url_for_any_ref(ref: str, monkeypatch: pytest.MonkeyPatch):
    from nat.tool.github_tools import GithubFilesGroupConfig
    from nat.tool.github_tools import github_files_tool

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="first line\nsecond line\n")

    real_async_client = httpx.AsyncClient

    def async_client_with_mock_transport(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real_async_client(*args, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", async_client_with_mock_transport)

    async with github_files_tool(GithubFilesGroupConfig(), MagicMock()) as function_info:
        result = await function_info.single_fn(f"https://github.com/owner/repo/blob/{ref}/docs/README.md#L2")

    assert result == "```second line\n```"
    assert len(requests) == 1
    # A refs/heads/ prefix only matches branch names, so commit and tag links would get a 404.
    assert str(requests[0].url) == f"https://raw.githubusercontent.com/owner/repo/{ref}/docs/README.md"


@pytest.mark.parametrize("since", ["yesterday", "2026-10-01", "2026-10-01 00:00:00", "2026-1-1T0:0:0Z", 20261001])
def test_get_issue_rejects_since_values_that_are_not_iso_8601_timestamps(since: str | int):
    from nat.tool.github_tools import GithubGetIssueModel

    with pytest.raises(ValidationError):
        GithubGetIssueModel(since=since)


@pytest.mark.parametrize("since", [None, "2026-10-01T00:00:00Z"])
def test_get_issue_keeps_valid_since_values(since: str | None):
    from nat.tool.github_tools import GithubGetIssueModel

    assert GithubGetIssueModel(since=since).since == since
