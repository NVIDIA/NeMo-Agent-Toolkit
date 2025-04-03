#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import datetime
import functools
import json
import logging
import os
import re
import subprocess
from collections.abc import Callable

logger = logging.getLogger(__name__)


def _run_cmd(exe: str, *args: str):
    """Runs a command with args and returns its output"""

    cmd_list = (exe, ) + args

    # Join the args to make the command string (for logging only)
    cmd_str = " ".join(cmd_list)

    # If we only passed in one executable (could be piping commands together) then use a shell
    shell = len(args) <= 0

    if (shell):
        # For logging purposes, if we only passed in one executable, then clear the exe name to make logging better
        exe = ""

    try:
        ret = subprocess.check_output(cmd_list, stderr=subprocess.PIPE, shell=shell)
        output = ret.decode("UTF-8").rstrip("\n")

        logger.debug("Running %s command: `%s`, Output: '%s'", exe, cmd_str, output)

        return output
    except subprocess.CalledProcessError as e:
        logger.warning("Running %s command [ERRORED]: `%s`, Output: '%s'",
                       exe,
                       cmd_str,
                       e.stderr.decode("UTF-8").rstrip("\n"))
        raise


def _gh(*args):
    """Runs a Github CLI command and returns its output"""

    return _run_cmd("gh", *args)


def _git(*args):
    """Runs a git command and returns its output"""
    return _run_cmd("git", *args)


class GitWrapper:

    @functools.lru_cache
    @staticmethod
    def get_closest_tag():
        """
        Determines the version of the repo by using `git describe`

        Returns
        -------
        str
            The full version of the repo in the format 'v#.#.#{a|b|rc}'
        """
        return _git("describe", "--tags", "--abbrev=0")

    @functools.lru_cache
    @staticmethod
    def get_repo_version():
        """
        Determines the version of the repo using `git describe` and returns only
        the major and minor portion

        Returns
        -------
        str
            The partial version of the repo in the format '{major}.{minor}'
        """

        full_repo_version = GitWrapper.get_closest_tag()

        match = re.match(r"^v?(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?", full_repo_version)

        if (match is None):
            logger.debug("Could not determine repo major minor version. Full repo version: %s.", full_repo_version)
            return None

        out_version = match.group("major")

        if (match.group("minor")):
            out_version += "." + match.group("minor")

        return out_version

    @functools.lru_cache
    @staticmethod
    def get_repo_owner_name():

        # pylint: disable=anomalous-backslash-in-string
        return "NVIDIA/" + _run_cmd("git remote -v | grep -oP '/\\K\\w*(?=\\.git \\(fetch\\))' | head -1")

    @functools.lru_cache
    @staticmethod
    def get_repo_remote_name(repo_owner_and_name: str):

        return _run_cmd(f"git remote -v | grep :{repo_owner_and_name} | grep \"(fetch)\" | head -1 | cut -f1")

    @functools.lru_cache
    @staticmethod
    def is_ref_valid(git_ref: str):

        try:
            return _git("rev-parse", "--verify", git_ref) != ""
        except subprocess.CalledProcessError:
            return False

    @functools.lru_cache
    @staticmethod
    def get_remote_branch(local_branch_ref: str, *, repo_owner_and_name: str | None = None):

        if (repo_owner_and_name is None):
            repo_owner_and_name = GitWrapper.get_repo_owner_name()

        remote_name = GitWrapper.get_repo_remote_name(repo_owner_and_name)

        remote_branch_ref = f"{remote_name}/{local_branch_ref}"

        if (GitWrapper.is_ref_valid(remote_branch_ref)):
            return remote_branch_ref

        logger.info("Remote branch '%s' for repo '%s' does not exist. Falling back to rev-parse",
                    remote_branch_ref,
                    repo_owner_and_name)

        remote_branch_ref = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", local_branch_ref + "@{upstream}")

        return remote_branch_ref

    @functools.lru_cache
    @staticmethod
    def get_target_remote_branch():
        base_ref = os.environ.get("CI_MERGE_REQUEST_TARGET_BRANCH_NAME")
        if (base_ref is not None):
            return base_ref

        try:
            base_ref = "develop"

            # If our current branch and the base ref are the same, then use main
            if (base_ref == GitWrapper.get_current_branch()):
                logger.warning("Current branch is the same as the tagged branch: %s. Falling back to 'main'", base_ref)
                base_ref = "develop"

        except Exception:
            logger.exception("Could not determine branch version falling back to develop", exc_info=True)
            base_ref = "develop"

        return GitWrapper.get_remote_branch(base_ref)

    @functools.lru_cache
    @staticmethod
    def get_repo_dir():
        """
        Returns the top level directory for this git repo
        """
        return _git("rev-parse", "--show-toplevel")

    @functools.lru_cache
    @staticmethod
    def get_current_branch():
        """Returns the name of the current branch"""
        name = _git("rev-parse", "--abbrev-ref", "HEAD")
        name = name.rstrip()
        return name

    @staticmethod
    def add_files(*files_to_add):
        """Runs git add on file"""
        return _git("add", *files_to_add)

    @functools.lru_cache
    @staticmethod
    def get_file_add_date(file_path):
        """Return the date a given file was added to git"""
        date_str = _run_cmd(f"git log --follow --format=%as -- {file_path} | tail -n 1")
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")

    @staticmethod
    def get_uncommitted_files():
        """
        Returns a list of all changed files that are not yet committed. This
        means both untracked/unstaged as well as uncommitted files too.
        """
        files = _git("status", "-u", "-s")
        ret = []
        for f in files.splitlines():
            f = f.strip(" ")
            f = re.sub(r"\s+", " ", f)  # noqa: W605
            tmp = f.split(" ", 1)
            # only consider staged files or uncommitted files
            # in other words, ignore untracked files
            if tmp[0] == "M" or tmp[0] == "A":
                ret.append(tmp[1])
        return ret

    @staticmethod
    def diff(target_ref: str, base_ref: str, merge_base: bool = False, staged: bool = False):

        assert base_ref is not None or base_ref != "", "base_ref must be a valid ref"
        assert target_ref is not None or target_ref != "", "target_ref must be a valid ref"

        args = ["--no-pager", "diff", "--name-only", "--ignore-submodules"]

        if (merge_base):
            args.append("--merge-base")

        if (staged):
            args.append("--cached")

        args += [target_ref, base_ref]

        return _git(*args).splitlines()

    @staticmethod
    def diff_index(target_ref: str, merge_base: bool = False, staged: bool = False):

        assert target_ref is not None or target_ref != "", "target_ref must be a valid ref"

        args = ["--no-pager", "diff-index", "--name-only", "--ignore-submodules"]

        if (merge_base):
            args.append("--merge-base")

        if (staged):
            args.append("--cached")

        args += [target_ref]

        return _git(*args).splitlines()

    @staticmethod
    def merge_base(target_ref: str, base_ref: str = "HEAD"):

        assert base_ref is not None or base_ref != "", "base_ref must be a valid ref"
        assert target_ref is not None or target_ref != "", "target_ref must be a valid ref"

        return _git("merge-base", target_ref, base_ref)


class GithubWrapper:

    @functools.lru_cache
    @staticmethod
    def has_cli():
        if os.environ.get("AIQ_AVOID_GH_CLI") is not None:
            logger.debug("AIQ_AVOID_GH_CLI is set. Skipping Github CLI check")
            return False
        try:
            _gh("--version")

            # Run a test function
            repo_name = _gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")

            logger.debug("Github CLI is installed. Using repo: %s", repo_name)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.debug("Github CLI is not installed")
            return False

    @functools.lru_cache
    @staticmethod
    def get_repo_owner_name():

        # Make sure we have the CLI
        if (not GithubWrapper.has_cli()):
            return None

        return _gh("repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner")

    @functools.lru_cache
    @staticmethod
    def get_pr_info() -> dict | None:

        # Make sure we have the CLI
        if (not GithubWrapper.has_cli()):
            return None

        # List of fields to get from the PR
        fields = [
            "baseRefName",
            "number",
        ]

        json_output = _gh("pr", "status", "--json", ",".join(fields), "--jq", ".currentBranch")

        if (json_output == ""):
            return None

        return json.loads(json_output)

    @functools.lru_cache
    @staticmethod
    def is_pr():

        return GithubWrapper.get_pr_info() is not None

    @functools.lru_cache
    @staticmethod
    def get_pr_number():

        pr_info = GithubWrapper.get_pr_info()

        if (pr_info is None):
            return None

        return pr_info["number"]

    @functools.lru_cache
    @staticmethod
    def get_pr_base_ref_name():

        pr_info = GithubWrapper.get_pr_info()

        if (pr_info is None):
            return None

        return pr_info["baseRefName"]

    @functools.lru_cache
    @staticmethod
    def get_pr_target_remote_branch():

        # Make sure we are in a PR
        if (not GithubWrapper.is_pr()):
            return None

        # Get the PR base reference
        base_ref = GithubWrapper.get_pr_base_ref_name()

        # Now determine the remote ref name matching our repository
        remote_name = GitWrapper.get_remote_branch(base_ref, repo_owner_and_name=GithubWrapper.get_repo_owner_name())

        return remote_name


def _is_repo_relative(f: str, git_root: str = None):
    if (git_root is None):
        git_root = GitWrapper.get_repo_dir()

    abs_f = os.path.abspath(f)

    rel_path = os.path.relpath(abs_f, git_root)

    return not rel_path.startswith("../")


def get_merge_target():
    """
    Returns the merge target branch for the current branch as if it were a PR/MR

    Order of operations:
    1. Try to determine the target branch from GitLab CI (assuming were in a PR)
    2. Try to guess the target branch as "branch-<major>.<minor>" using the most recent tag (assuming we have a remote
       pointing to the base repo)
    3. Try to determine the target branch by finding a head reference that matches "branch-*" and is in this history
    4. Fall back to "main" if all else fails or the target branch and current branch are the same

    Returns
    -------
    str
        Ref name of the target branch
    """
    #

    remote_branch = GithubWrapper.get_pr_target_remote_branch()

    if (remote_branch is None):
        # Try to use tags
        remote_branch = GitWrapper.get_target_remote_branch()

    if (remote_branch is None):

        raise RuntimeError("Could not determine remote_branch. Manually set TARGET_BRANCH to continue")

    return remote_branch


def determine_merge_commit(current_branch="HEAD"):
    """
    When running outside of CI, this will estimate the target merge commit hash of `current_branch` by finding a common
    ancester with the remote branch 'branch-{major}.{minor}' where {major} and {minor} are determined from the repo
    version.

    Parameters
    ----------
    current_branch : str, optional
        Which branch to consider as the current branch, by default "HEAD"

    Returns
    -------
    str
        The common commit hash ID
    """

    remote_branch = get_merge_target()

    common_commit = GitWrapper.merge_base(remote_branch, current_branch)

    logger.info("Determined TARGET_BRANCH as: '%s'. With merge-commit: %s", remote_branch, common_commit)

    return common_commit


def filter_files(files: str | list[str], path_filter: Callable[[str], bool] = None) -> list[str]:
    """
    Filters out the input files according to a predicate

    Parameters
    ----------
    files : typing.Union[str, list[str]]
        List of files to filter
    path_filter : Callable[[str], bool], optional
        Predicate that returns True/False for each file, by default None

    Returns
    -------
    list[str]
        Filtered list of files
    """

    # Convert all to array of strings
    if (isinstance(files, str)):
        files = files.splitlines()

    git_root = GitWrapper.get_repo_dir()

    ret_files: list[str] = []

    for file in files:
        # Check that we are relative to the git repo
        assert _is_repo_relative(file, git_root=git_root), f"Path {file} must be relative to git root: {git_root}"

        if (path_filter is None or path_filter(file)):
            ret_files.append(file)

    return ret_files


def changed_files(target_ref: str = None,
                  base_ref="HEAD",
                  *,
                  merge_base: bool = True,
                  staged=False,
                  path_filter: Callable[[str], bool] = None):
    """
    Comparison between 2 commits in the repo. Returns a list of files that have been filtered by `path_filter`

    Parameters
    ----------
    target_ref : str, optional
        The branch name to use as the target. If set to None, it will use the value in $TARGET_BRANCH
    base_ref : str, optional
        The base branch name, by default "HEAD"
    merge_base : bool, optional
        Setting this to True will calculate the diff to the merge-base between `taget_ref` and `base_ref`. Setting to
        False will compre the HEAD of each ref
    staged : bool, optional
        Whether or not to include staged, but not committed, files, by default False
    path_filter : Callable[[str], bool], optional
        A predicate to apply to the list of files, by default None

    Returns
    -------
    list[str]
        The list of files that have changed between the refs filtered by `path_filter`
    """

    if (target_ref is None):
        target_ref = os.environ.get("TARGET_BRANCH", None)

    if (target_ref is None):
        target_ref = get_merge_target()

    logger.info("Comparing %s..%s with merge_base: %s, staged: %s", target_ref, base_ref, merge_base, staged)

    diffs = GitWrapper.diff(target_ref, base_ref, merge_base=merge_base, staged=staged)

    return filter_files(diffs, path_filter=path_filter)


def modified_files(target_ref: str = None,
                   *,
                   merge_base: bool = True,
                   staged=False,
                   path_filter: Callable[[str], bool] = None):
    """
    Comparison between the working tree and a target branch. Returns a list of files that have been filtered by
    `path_filter`

    Parameters
    ----------
    target_ref : str, optional
        The branch name to use as the target. If set to None, it will use the value in $TARGET_BRANCH
    merge_base : bool, optional
        Setting this to True will calculate the diff to the merge-base between `taget_ref` and `base_ref`. Setting to
        False will compre the HEAD of each ref
    staged : bool, optional
        Whether or not to include staged, but not committed, files, by default False
    path_filter : Callable[[str], bool], optional
        A predicate to apply to the list of files, by default None

    Returns
    -------
    list[str]
        The list of files that have changed between the refs filtered by `path_filter`
    """

    if (target_ref is None):
        target_ref = os.environ.get("TARGET_BRANCH", None)

    if (target_ref is None):
        target_ref = get_merge_target()

    logger.info("Comparing index to %s with merge_base: %s, staged: %s", target_ref, merge_base, staged)

    diffs = GitWrapper.diff_index(target_ref, merge_base=merge_base, staged=staged)

    return filter_files(diffs, path_filter=path_filter)


def staged_files(base_ref="HEAD", *, path_filter: Callable[[str], bool] = None):
    """
    Calculates the different between the working tree and the index including staged files. Returns a list of files that
    have been filtered by `path_filter`.

    Identical to `modified_files` with `staged=True`

    Parameters
    ----------
    base_ref : str, optional
        The base branch name, by default "HEAD"
    path_filter : Callable[[str], bool], optional
        A predicate to apply to the list of files, by default None

    Returns
    -------
    list[str]
        The list of files that have changed between the refs filtered by `path_filter`
    """

    return modified_files(target_ref=base_ref, merge_base=False, staged=True, path_filter=path_filter)


def all_files(*paths, base_ref="HEAD", path_filter: Callable[[str], bool] = None):
    """
    Returns a list of all files in the repo that have been filtered by `path_filter`.

    Parameters
    ----------
    paths : list[str]
        The list of paths to include in the search
    base_ref : str, optional
        The base branch name, by default "HEAD"
    path_filter : Callable[[str], bool], optional
        A predicate to apply to the list of files, by default None

    Returns
    -------
    list[str]
        The list of files in the repo filtered by `path_filter`
    """

    git_args = ["ls-tree", "-r", "--name-only", base_ref] + list(paths)

    ls_files = _git(*git_args)

    return filter_files(ls_files, path_filter=path_filter)


def add_files(*files_to_add):
    """
    Calls `git add` on the input files

    Returns
    -------
    str
        Output of the git command
    """
    return GitWrapper.add_files(*files_to_add)


def get_file_add_date(filename: str):
    """
    Returns the date a given file was added to git.

    Parameters
    ----------
    filename : str
        Filename in question

    Returns
    -------
    datetime.datetime
        Time the file was added.
    """

    return GitWrapper.get_file_add_date(filename)


def _parse_args():
    argparser = argparse.ArgumentParser("Executes a gitutil action")
    argparser.add_argument("action", choices=['get_merge_target'], help="Action to execute")
    argparser.add_argument('--current-branch', default="HEAD", help='Name of current branch')
    args = argparser.parse_args()
    return args


def _main():
    log_level = logging.getLevelName(os.environ.get("AIQ_LOG_LEVEL", "INFO"))
    logging.basicConfig(format="%(levelname)s:%(message)s", level=log_level)

    args = _parse_args()

    if args.action == 'get_merge_target':
        print(determine_merge_commit(current_branch=args.current_branch))


if __name__ == '__main__':
    _main()
