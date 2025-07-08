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

import logging
from dataclasses import dataclass
from pathlib import Path

from git import Repo

logger = logging.getLogger(__name__)


@dataclass
class RepoContext:
    """Context manager for repository operations."""
    repo_url: str
    base_path: Path
    repo: Repo | None = None

    def __post_init__(self):
        self.repo_name = self.repo_url.split('/')[-1].replace('.git', '')
        self.repo_path = self.base_path / self.repo_name


class RepoManager:

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.active_repos = {}

    async def setup_repository(self, repo_url: str, base_commit: str) -> RepoContext:
        """Setup a repository at a specific commit."""
        repo_path = get_repo_path(str(self.workspace), repo_url)

        if str(repo_path) in self.active_repos:
            context = self.active_repos[str(repo_path)]
            await checkout_commit(context.repo, base_commit)
            return context

        repo = await clone_repository(repo_url, repo_path)
        await checkout_commit(repo, base_commit)

        context = RepoContext(repo_url=repo_url, base_path=self.workspace, repo=repo)
        self.active_repos[str(repo_path)] = context
        return context

    async def cleanup(self):
        """Clean up all managed repositories."""
        import shutil
        for repo_path_str in list(self.active_repos.keys()):
            repo_path = Path(repo_path_str)
            if repo_path.exists():
                shutil.rmtree(repo_path)
        self.active_repos.clear()


def get_repo_path(workspace_dir: str, repo_url: str) -> Path:
    """Generate a unique path for the repository."""
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    return Path(workspace_dir) / repo_name


async def clone_repository(repo_url: str, target_path: Path) -> Repo:
    """Clone a repository to the specified path."""
    logger.info("Cloning repository %s to %s", repo_url, target_path)
    return Repo.clone_from(repo_url, target_path)


async def checkout_commit(repo: Repo, commit_hash: str):
    """Checkout a specific commit in the repository."""
    logger.info("Checking out commit %s", commit_hash)
    repo.git.checkout(commit_hash)
