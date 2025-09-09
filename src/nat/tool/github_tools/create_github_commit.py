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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GithubCommitCodeModel(BaseModel):
    branch: str = Field(description="The branch of the remote repo to which the code will be committed")
    commit_msg: str = Field(description="Message with which the code will be committed to the remote repo")
    local_path: str = Field(description="Local filepath of the file that has been updated and "
                            "needs to be committed to the remote repo")
    remote_path: str = Field(description="Remote filepath of the updated file in GitHub. Path is relative to "
                             "root of current repository")


class GithubCommitCodeModelList(BaseModel):
    updated_files: list[GithubCommitCodeModel] = Field(description=("A list of local filepaths and commit messages"))


class GithubCommitCodeConfig(FunctionBaseConfig, name="github_commit_code_tool"):
    """
    Tool that commits and pushes modified code to a remote GitHub repository asynchronously.
    """
    repo_name: str = Field(description="The repository name in the format 'owner/repo'")
    local_repo_dir: str = Field(description="Absolute path to the root of the repo, cloned locally")
    timeout: int = Field(default=300, description="The timeout configuration to use when sending requests.")


@register_function(config_type=GithubCommitCodeConfig)
async def commit_code_async(config: GithubCommitCodeConfig, builder: Builder):
    """
    Commits and pushes modified code to a remote GitHub repository asynchronously.

    """
    import json
    import os

    import httpx

    github_pat = os.getenv("GITHUB_PAT")
    if not github_pat:
        raise ValueError("GITHUB_PAT environment variable must be set")

    # define the headers for the payload request
    headers = {"Authorization": f"Bearer {github_pat}", "Accept": "application/vnd.github+json"}

    async def _github_commit_code(updated_files) -> list:
        results = []
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            for file_ in updated_files:
                branch = file_.branch
                commit_msg = file_.commit_msg
                local_path = file_.local_path
                remote_path = file_.remote_path

                # Read content from the local file
                local_path = os.path.join(config.local_repo_dir, local_path)
                with open(local_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # Step 1. Create a blob with the updated contents of the file
                blob_url = f'https://api.github.com/repos/{config.repo_name}/git/blobs'
                blob_data = {'content': content, 'encoding': 'utf-8'}
                blob_response = await client.request("POST", blob_url, json=blob_data, headers=headers)
                blob_response.raise_for_status()
                blob_sha = blob_response.json()['sha']

                # Step 2: Get the base tree SHA. The commit will be pushed to this ref node in the Git graph
                ref_url = f'https://api.github.com/repos/{config.repo_name}/git/refs/heads/{branch}'
                ref_response = await client.request("GET", ref_url, headers=headers)
                ref_response.raise_for_status()
                base_tree_sha = ref_response.json()['object']['sha']

                # Step 3. Create an updated tree (Git graph) with the new blob
                tree_url = f'https://api.github.com/repos/{config.repo_name}/git/trees'
                tree_data = {
                    'base_tree': base_tree_sha,
                    'tree': [{
                        'path': remote_path, 'mode': '100644', 'type': 'blob', 'sha': blob_sha
                    }]
                }
                tree_response = await client.request("POST", tree_url, json=tree_data, headers=headers)
                tree_response.raise_for_status()
                tree_sha = tree_response.json()['sha']

                # Step 4: Create a commit
                commit_url = f'https://api.github.com/repos/{config.repo_name}/git/commits'
                commit_data = {'message': commit_msg, 'tree': tree_sha, 'parents': [base_tree_sha]}
                commit_response = await client.request("POST", commit_url, json=commit_data, headers=headers)
                commit_response.raise_for_status()
                commit_sha = commit_response.json()['sha']

                # Step 5: Update the reference in the Git graph
                update_ref_url = f'https://api.github.com/repos/{config.repo_name}/git/refs/heads/{branch}'
                update_ref_data = {'sha': commit_sha}
                update_ref_response = await client.request("PATCH",
                                                           update_ref_url,
                                                           json=update_ref_data,
                                                           headers=headers)
                update_ref_response.raise_for_status()

                payload_responses = {
                    'blob_resp': blob_response.json(),
                    'original_tree_ref': tree_response.json(),
                    'commit_resp': commit_response.json(),
                    'updated_tree_ref_resp': update_ref_response.json()
                }
                results.append(payload_responses)

        return json.dumps(results)

    yield FunctionInfo.from_fn(_github_commit_code,
                               description=(f"Commits and pushes modified code to a "
                                            f"GitHub repository in the repo named {config.repo_name}"),
                               input_schema=GithubCommitCodeModelList)
