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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class GithubListPullsModel(BaseModel):
    state: Literal["open", "closed", "all"] | None = Field('open', description="Issue state used in issue query filter")
    head: str | None = Field(None, description="Filters pulls by head user or head organization and branch name")
    base: str | None = Field(None, description="Filters pull by branch name")


class GithubListPullsModelList(BaseModel):
    filter_params: GithubListPullsModel = Field(description=("A list of query params when fetching pull requests "
                                                             "each of type GithubListPRModel"))


class GithubListPullsToolConfig(FunctionBaseConfig, name="github_list_pulls_tool"):
    """
    Tool that lists GitHub Pull Requests based on various filter parameters
    """
    repo_name: str = Field(description="The repository name in the format 'owner/repo'")
    timeout: int = Field(default=300, description="The timeout configuration to use when sending requests.")


@register_function(config_type=GithubListPullsToolConfig)
async def list_github_pulls_async(config: GithubListPullsToolConfig, builder: Builder):
    """
    Lists GitHub Pull Requests based on various filter parameters

    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    url = f"https://api.github.com/repos/{config.repo_name}/pulls"

    # define the headers for the payload request
    headers = {"Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"}

    async def _github_list_pulls(filter_params) -> dict:
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

    yield FunctionInfo.from_fn(_github_list_pulls,
                               description=(f"Lists GitHub PRs based on filter params "
                                            f"in the repo named {config.repo_name}"),
                               input_schema=GithubListPullsModelList)


class GithubGetPullModel(BaseModel):
    pull_number: str = Field(description="The number of the pull request that needs to be fetched")


class GithubGetPullToolConfig(FunctionBaseConfig, name="github_get_pull_tool"):
    """
    Tool that fetches a particular pull request in a GitHub repository asynchronously.
    """
    repo_name: str = "The repository name in the format 'owner/repo'"
    timeout: int = 300


@register_function(config_type=GithubGetPullToolConfig)
async def get_github_pull_async(config: GithubGetPullToolConfig, builder: Builder):
    """
    Fetches a particular pull request in a GitHub repository asynchronously.

    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    url = f"https://api.github.com/repos/{config.repo_name}/pulls"

    # define the headers for the payload request
    headers = {"Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"}

    async def _github_get_pull(pull_number) -> list:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            # update the url with the pull number that needs to be updated
            pull_url = os.path.join(url, pull_number)

            response = await client.request("GET", pull_url, headers=headers)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse and return the response JSON
            try:
                result = response.json()

            except ValueError as e:
                raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(result)

    yield FunctionInfo.from_fn(_github_get_pull,
                               description=(f"Fetches a particular GitHub pull request "
                                            f"in the repo named {config.repo_name}"),
                               input_schema=GithubGetPullModel)


class GithubGetPullCommitsToolConfig(FunctionBaseConfig, name="github_get_pull_commits_tool"):
    """
    Configuration for the GitHub Get Pull Commits Tool.
    """
    repo_name: str = "The repository name in the format 'owner/repo'"
    timeout: int = 300


@register_function(config_type=GithubGetPullCommitsToolConfig)
async def get_github_pull_commits_async(config: GithubGetPullCommitsToolConfig, builder: Builder):
    """
    Fetches the commits associated with a particular pull request in a GitHub repository asynchronously.

    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    url = f"https://api.github.com/repos/{config.repo_name}/pulls"

    # define the headers for the payload request
    headers = {"Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"}

    async def _github_get_pull(pull_number) -> list:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            # update the url with the pull number that needs to be updated
            pull_url = os.path.join(url, pull_number)
            pull_commits_url = os.path.join(pull_url, "commits")

            response = await client.request("GET", pull_commits_url, headers=headers)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse and return the response JSON
            try:
                result = response.json()

            except ValueError as e:
                raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(result)

    yield FunctionInfo.from_fn(_github_get_pull,
                               description=("Fetches the commits for a particular GitHub pull request "
                                            f" in the repo named {config.repo_name}"),
                               input_schema=GithubGetPullModel)


class GithubGetPullFilesToolConfig(FunctionBaseConfig, name="github_get_pull_files_tool"):
    """
    Configuration for the GitHub Get Pull Files Tool.
    """
    repo_name: str = "The repository name in the format 'owner/repo'"
    timeout: int = 300


@register_function(config_type=GithubGetPullFilesToolConfig)
async def get_github_pull_files_async(config: GithubGetPullFilesToolConfig, builder: Builder):
    """
    Fetches the files associated with a particular pull request in a GitHub repository asynchronously.

    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    url = f"https://api.github.com/repos/{config.repo_name}/pulls"

    # define the headers for the payload request
    headers = {"Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"}

    async def _github_get_pull(pull_number) -> list:
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            # update the url with the pull number that needs to be updated
            pull_url = os.path.join(url, pull_number)
            pull_files_url = os.path.join(pull_url, "files")

            response = await client.request("GET", pull_files_url, headers=headers)

            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Parse and return the response JSON
            try:
                result = response.json()

            except ValueError as e:
                raise ValueError("The API response is not valid JSON.") from e

        return json.dumps(result)

    yield FunctionInfo.from_fn(_github_get_pull,
                               description=("Fetches the files for a particular GitHub pull request "
                                            f" in the repo named {config.repo_name}"),
                               input_schema=GithubGetPullModel)
