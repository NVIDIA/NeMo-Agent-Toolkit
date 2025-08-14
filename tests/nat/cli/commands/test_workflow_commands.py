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

from pathlib import Path
from unittest.mock import patch

from nat.cli.commands.workflow.workflow_commands import _get_nat_dependency
from nat.cli.commands.workflow.workflow_commands import get_repo_root


def test_get_repo_root(project_dir: str):
    assert get_repo_root() == Path(project_dir)


@patch('nat.cli.entrypoint.get_version')
def test_get_nat_dependency_non_editable_with_version(mock_get_version):
    """Test that non-editable mode with valid version returns versioned dependency."""
    mock_get_version.return_value = "1.2.3"

    result = _get_nat_dependency(editable=False)
    assert result == "nvidia-nat[langchain]~=1.2"


@patch('nat.cli.entrypoint.get_version')
def test_get_nat_dependency_non_editable_unknown_version(mock_get_version):
    """Test that non-editable mode with unknown version returns unversioned dependency."""
    mock_get_version.return_value = "unknown"

    result = _get_nat_dependency(editable=False)
    assert result == "nvidia-nat[langchain]"
