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

from typing import Literal

from pydantic import BaseModel
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GithubUpdateIssueModel(BaseModel):
    issue_number: str = Field(description="The issue number that will be updated")
    title: str | None = Field(None, description="The title of the GitHub Issue")
    body: str | None = Field(None, description="The body of the GitHub Issue")
    state: Literal["open", "closed"] | None = Field(None, description="The new state of the issue")

    state_reason: Literal["completed", "not_planned", "reopened", None] | None = Field(
        None, description="The reason for changing the state of the issue")

    labels: list[str] | None = Field(None, description="A list of labels to assign to the issue")
    assignees: list[str] | None = Field(None, description="A list of assignees to assign to the issue")


class GithubUpdateIssueModelList(BaseModel):
    issues: list[GithubUpdateIssueModel] = Field(description=("A list of GitHub issues each "
                                                              "of type GithubUpdateIssueModel"))


class GithubUpdateIssueToolConfig(FunctionBaseConfig, name="github_update_issue_tool"):
    """
    Tool that updates an issue in a GitHub repository asynchronously.
    """
    repo_name: str = "The repository name in the format 'owner/repo'"
    timeout: int = 300


@register_function(config_type=GithubUpdateIssueToolConfig)
async def update_github_issue_async(config: GithubUpdateIssueToolConfig, builder: Builder):
    """
    Updates an issue in a GitHub repository asynchronously.
    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    url = f"https://api.github.com/repos/{config.repo_name}/issues"

    # define the headers for the payload request
    headers = {"Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"}

    async def _github_update_issue(issues) -> list:
        results = []
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            for issue in issues:
                payload = issue.dict(exclude_unset=True)

                # update the url with the issue number that needs to be updated
                issue_number = payload.pop("issue_number")
                issue_url = os.path.join(url, issue_number)

                response = await client.request("PATCH", issue_url, json=payload, headers=headers)

                # Raise an exception for HTTP errors
                response.raise_for_status()

                # Parse and return the response JSON
                try:
                    result = response.json()
                    results.append(result)

                except ValueError as e:
                    raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(results)

    yield FunctionInfo.from_fn(_github_update_issue,
                               description=(f"Updates a GitHub issue in the "
                                            f"repo named {config.repo_name}"),
                               input_schema=GithubUpdateIssueModelList)
