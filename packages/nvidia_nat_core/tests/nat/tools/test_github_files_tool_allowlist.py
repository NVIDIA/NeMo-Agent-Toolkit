# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Tests for the github_files_tool allowed_repos allowlist.

Direct unit coverage of the _repo_path_is_allowed helper plus the
GithubFilesGroupConfig shape. End-to-end test_tool coverage of the get()
closure is more involved (requires a mock httpx transport) and is out of
scope for this defensive hardening PR.
"""

import pytest

from nat.tool.github_tools import GithubFilesGroupConfig
from nat.tool.github_tools import _repo_path_is_allowed


class TestRepoPathAllowlist:
    """_repo_path_is_allowed should pass with no allowlist and enforce scope when set."""

    def test_none_allowlist_permits_everything(self):
        """Backward compat: allowed_repos=None keeps the current permissive default."""
        assert _repo_path_is_allowed("NVIDIA/NeMo-Agent-Toolkit", None) is True
        assert _repo_path_is_allowed("any-org/any-repo", None) is True

    @pytest.mark.parametrize(
        "repo_path,allowlist,expected",
        [
            ("NVIDIA/NeMo-Agent-Toolkit", ["NVIDIA/NeMo-Agent-Toolkit"], True),
            ("NVIDIA/NeMo-Agent-Toolkit", ["NVIDIA/*"], True),
            ("nvidia/nemo-agent-toolkit", ["NVIDIA/*"], True),
            ("NVIDIA/Other-Repo", ["NVIDIA/*"], True),
            ("NVIDIA/Other-Repo", ["NVIDIA/NeMo-Agent-Toolkit"], False),
            ("attacker/exfil", ["NVIDIA/*"], False),
            ("attacker/exfil", ["NVIDIA/NeMo-Agent-Toolkit"], False),
        ],
    )
    def test_explicit_and_wildcard_matches(self, repo_path, allowlist, expected):
        assert _repo_path_is_allowed(repo_path, allowlist) is expected

    def test_case_insensitive_match(self):
        """Operator can write 'NVIDIA/Repo' and still match the lower-case URL form."""
        assert _repo_path_is_allowed("nvidia/repo", ["NVIDIA/Repo"]) is True
        assert _repo_path_is_allowed("NVIDIA/REPO", ["nvidia/repo"]) is True

    def test_multiple_entries_are_or(self):
        """Any one matching entry is sufficient to allow the repo."""
        allowlist = ["NVIDIA/*", "apache/spark", "custom-org/my-repo"]
        assert _repo_path_is_allowed("NVIDIA/x", allowlist) is True
        assert _repo_path_is_allowed("apache/spark", allowlist) is True
        assert _repo_path_is_allowed("custom-org/my-repo", allowlist) is True
        assert _repo_path_is_allowed("custom-org/OTHER", allowlist) is False

    @pytest.mark.parametrize(
        "bad_entry",
        [
            "",
            "no-slash",
            None,
            123,
            "/repo",           # empty org
            "org/",            # empty repo
            "org/repo/extra",  # three segments
            "a/b/c/d",         # many segments
            "//",              # all empty
            "/",               # single slash, two empty sides
        ],
    )
    def test_malformed_allowlist_entries_are_ignored(self, bad_entry):
        """A malformed entry mixed in with valid ones must not short-circuit to True."""
        allowlist = [bad_entry, "NVIDIA/*"]
        assert _repo_path_is_allowed("NVIDIA/toolkit", allowlist) is True
        assert _repo_path_is_allowed("attacker/exfil", allowlist) is False
        # The bad entry alone should not match anything.
        assert _repo_path_is_allowed("NVIDIA/toolkit", [bad_entry]) is False

    @pytest.mark.parametrize(
        "bad_path",
        [
            "",
            "orgonly",
            "/repo",
            "org/",
            "org/repo/extra",  # previously accepted by the 'contains /' check
            "a/b/c/d",         # previously accepted by the 'contains /' check
            "//",
            "/",
        ],
    )
    def test_malformed_repo_path_is_rejected(self, bad_path):
        """Tighter validation: repo_path must be exactly 'org/repo'.

        The earlier 'contains /' check accepted 'org/repo/extra' and 'org/'
        as partial matches against allowlist prefixes — a narrow but real
        scope-widening bug. Now split+segment-count enforces the intent.
        """
        assert _repo_path_is_allowed(bad_path, ["NVIDIA/*"]) is False

    def test_empty_allowlist_blocks_everything(self):
        """Explicit empty list means 'no repos allowed' — an explicit deny-all."""
        assert _repo_path_is_allowed("NVIDIA/toolkit", []) is False
        assert _repo_path_is_allowed("any/any", []) is False


class TestGithubFilesGroupConfigShape:
    """Config must expose the allowed_repos field and default it to None."""

    def test_default_allowed_repos_is_none(self):
        cfg = GithubFilesGroupConfig()
        assert cfg.allowed_repos is None

    def test_accepts_explicit_repo_list(self):
        cfg = GithubFilesGroupConfig(allowed_repos=["NVIDIA/NeMo-Agent-Toolkit"])
        assert cfg.allowed_repos == ["NVIDIA/NeMo-Agent-Toolkit"]

    def test_accepts_wildcard_entry(self):
        cfg = GithubFilesGroupConfig(allowed_repos=["NVIDIA/*"])
        assert cfg.allowed_repos == ["NVIDIA/*"]

    def test_accepts_empty_list_deny_all(self):
        cfg = GithubFilesGroupConfig(allowed_repos=[])
        assert cfg.allowed_repos == []
