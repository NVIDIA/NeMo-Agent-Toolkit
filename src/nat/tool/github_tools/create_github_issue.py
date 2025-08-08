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

from pydantic import BaseModel
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class GithubCreateIssueModel(BaseModel):
    title: str = Field(description="The title of the GitHub Issue")
    body: str = Field(description="The body of the GitHub Issue")


class GithubCreateIssueModelList(BaseModel):
    issues: list[GithubCreateIssueModel] = Field(description=("A list of GitHub issues, "
                                                              "each with a title and a body"))


class GithubCreateIssueToolConfig(FunctionBaseConfig, name="github_create_issue_tool"):
    """
    Tool that creates an issue in a GitHub repository asynchronously.
    """
    repo_name: str = Field(description="The repository name in the format 'owner/repo'")
    timeout: int = Field(default=300, description="The timeout configuration to use when sending requests.")


@register_function(config_type=GithubCreateIssueToolConfig)
async def create_github_issue_async(config: GithubCreateIssueToolConfig, builder: Builder):
    """
    Creates an issue in a GitHub repository asynchronously.
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

    async def _github_post_issue(issues) -> list:
        results = []
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            for issue in issues:
                # define the payload body
                payload = issue.dict(exclude_unset=True)

                response = await client.request("POST", url, json=payload, headers=headers)

                # Raise an exception for HTTP errors
                response.raise_for_status()

                # Parse and return the response JSON
                try:
                    result = response.json()
                    results.append(result)

                except ValueError as e:
                    raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(results)

    yield FunctionInfo.from_fn(_github_post_issue,
                               description=(f"Creates a GitHub issue in the "
                                            f"repo named {config.repo_name}"),
                               input_schema=GithubCreateIssueModelList)
