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

from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GithubListIssueModel(BaseModel):
    state: Literal["open", "closed", "all"] | None = Field('open', description="Issue state used in issue query filter")
    assignee: str | None = Field("*", description="Assignee name used in issue query filter")
    creator: str | None = Field(None, description="Creator name used in issue query filter")
    mentioned: str | None = Field(None, description="Name of person mentioned in issue")
    labels: list[str] | None = Field(None, description="A list of labels that are assigned to the issue")
    since: str | None = Field(None, description="Only show results that were last updated after the given time.")

    @classmethod
    @field_validator('since', mode='before')
    def validate_since(cls, v):
        if v is None:
            return v
        try:
            # Parse the string to a datetime object
            parsed_date = datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
            # Return the formatted string
            return parsed_date.isoformat() + 'Z'
        except ValueError as e:
            raise ValueError("since must be in ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ") from e


class GithubListIssueModelList(BaseModel):
    filter_params: GithubListIssueModel = Field(description=("A list of query params when fetching issues "
                                                             "each of type GithubListIssueModel"))


class GithubListIssueToolConfig(FunctionBaseConfig, name="github_list_issues_tool"):
    """
    Configuration for the GitHub List Issues Tool.
    """
    repo_name: str = Field(description="The repository name in the format 'owner/repo'")
    timeout: int = Field(default=300, description="The timeout configuration to use when sending requests.")


@register_function(config_type=GithubListIssueToolConfig)
async def list_github_issue_async(config: GithubListIssueToolConfig, builder: Builder):
    """
    Lists GitHub Issues based on various filter parameters

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

    async def _github_list_issues(filter_params) -> dict:
        async with httpx.AsyncClient(timeout=config.timeout) as client:

            filter_params = filter_params.dict(exclude_unset=True)

            # filter out None values that are explictly set in the request body.
            filter_params = {k: v for k, v in filter_params.items() if v is not None}

            response = await client.request("GET", url, params=filter_params, headers=headers)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse and return the response JSON
            try:
                result = response.json()

            except ValueError as e:
                raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(result)

    yield FunctionInfo.from_fn(_github_list_issues,
                               description=(f"Lists GitHub issues based on filter "
                                            f"params in the repo named {config.repo_name}"),
                               input_schema=GithubListIssueModelList)


class GithubGetIssueModel(BaseModel):
    issue_number: str = Field(description="The issue number that needs to be fetched")


class GithubGetIssueToolConfig(FunctionBaseConfig, name="github_get_issue_tool"):
    """
    Tool that fetches a particular issue in a GitHub repository asynchronously.
    """
    repo_name: str = "The repository name in the format 'owner/repo'"
    timeout: int = 300


@register_function(config_type=GithubGetIssueToolConfig)
async def get_github_issue_async(config: GithubGetIssueToolConfig, builder: Builder):
    """
    Fetches a particular issue in a GitHub repository asynchronously.

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

    async def _github_get_issue(issue_number) -> list:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            # update the url with the issue number that needs to be updated
            issue_url = os.path.join(url, issue_number)

            response = await client.request("GET", issue_url, headers=headers)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse and return the response JSON
            try:
                result = response.json()

            except ValueError as e:
                raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(result)

    yield FunctionInfo.from_fn(_github_get_issue,
                               description=(f"Fetches a particular GitHub issue "
                                            f"in the repo named {config.repo_name}"),
                               input_schema=GithubGetIssueModel)
