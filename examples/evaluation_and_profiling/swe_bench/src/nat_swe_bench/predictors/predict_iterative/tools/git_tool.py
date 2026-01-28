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

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from git import Repo

logger = logging.getLogger(__name__)


@dataclass
class RepoContext:
    """Context manager for repository operations."""
    repo_url: str
    repo_path: Path  # Actual path where the repo is cloned
    repo: Repo | None = None


class RepoManager:

    def __init__(self, workspace_dir: str):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.active_repos = {}

    async def setup_repository(
        self, repo_url: str, base_commit: str, instance_id: str | None = None
    ) -> RepoContext:
        """Setup a repository at a specific commit.

        Args:
            repo_url: URL of the repository to clone
            base_commit: Commit hash to checkout
            instance_id: Optional instance ID for workspace isolation. When provided,
                         each instance gets its own clean workspace directory.
        """
        repo_path = get_repo_path(str(self.workspace), repo_url, instance_id)

        if str(repo_path) in self.active_repos:
            context = self.active_repos[str(repo_path)]
            await checkout_commit(context.repo, base_commit)
            return context

        repo = await clone_repository(repo_url, repo_path)
        await checkout_commit(repo, base_commit)

        context = RepoContext(repo_url=repo_url, repo_path=repo_path, repo=repo)
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


def get_repo_path(workspace_dir: str, repo_url: str, instance_id: str | None = None) -> Path:
    """Generate a unique path for the repository.

    Args:
        workspace_dir: Base workspace directory
        repo_url: URL of the repository
        instance_id: Optional instance ID for unique workspace isolation

    Returns:
        Path to the repository. If instance_id is provided, returns
        workspace_dir/instance_id/org/repo for complete isolation.
        Otherwise returns workspace_dir/org/repo.
    """
    if "://" in repo_url:
        path = urlparse(repo_url).path
    else:
        # SSH form: git@host:org/repo.git
        path = repo_url.split(":", 1)[-1]
    parts = path.strip("/").split("/")
    repo_name = parts[-1].replace('.git', '')
    org_name = parts[-2]  # Organization name

    # If instance_id is provided, create isolated workspace per instance
    if instance_id:
        return Path(workspace_dir) / instance_id / org_name / repo_name

    # Default: workspace_dir/org/repo
    return Path(workspace_dir) / org_name / repo_name


async def clone_repository(repo_url: str, target_path: Path, timeout: int = 600) -> Repo:
    """Clone a repository with timeout and error handling.

    Args:
        repo_url: URL of the repository to clone.
        target_path: Local path to clone into.
        timeout: Maximum time in seconds for clone operation.

    Returns:
        The cloned Repo object.

    Raises:
        ValueError: If repo_url format is invalid.
        asyncio.TimeoutError: If clone exceeds timeout.
    """
    logger.info("Cloning repository %s to %s", repo_url, target_path)

    # Validate URL format
    if not (repo_url.startswith('https://') or repo_url.startswith('git@')):
        raise ValueError(f"Invalid repository URL: {repo_url}")

    # Clean existing path
    if target_path.exists():
        await asyncio.to_thread(shutil.rmtree, target_path)

    try:
        repo = await asyncio.wait_for(
            asyncio.to_thread(Repo.clone_from, repo_url, target_path),
            timeout=timeout
        )
        logger.info("Successfully cloned %s", repo_url)
        return repo
    except asyncio.TimeoutError:
        logger.error("Clone timed out for %s after %ds", repo_url, timeout)
        if target_path.exists():
            await asyncio.to_thread(shutil.rmtree, target_path)
        raise
    except Exception as e:
        logger.error("Clone failed for %s: %s", repo_url, e)
        if target_path.exists():
            await asyncio.to_thread(shutil.rmtree, target_path)
        raise


async def checkout_commit(repo: Repo, commit_hash: str, timeout: int = 120):
    """Checkout a specific commit with timeout and error handling.

    Args:
        repo: The repository object.
        commit_hash: The commit hash to checkout.
        timeout: Maximum time in seconds for checkout operation.

    Raises:
        asyncio.TimeoutError: If checkout exceeds timeout.
    """
    logger.info("Checking out commit %s", commit_hash)
    try:
        await asyncio.wait_for(
            asyncio.to_thread(repo.git.checkout, commit_hash),
            timeout=timeout
        )
        logger.info("Successfully checked out %s", commit_hash)
    except asyncio.TimeoutError:
        logger.error("Checkout timed out for %s after %ds", commit_hash, timeout)
        raise
    except Exception as e:
        logger.error("Checkout failed for %s: %s", commit_hash, e)
        raise
