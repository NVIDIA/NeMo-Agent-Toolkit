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
def test_create_workflow_with_valid_name(tmp_path):
    """Ensure CLI succeeds with a valid workflow name."""
    runner = CliRunner()
    result = runner.invoke(
        create_command,
        ["my-workflow", "--no-install", "--workflow-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "Workflow 'my-workflow' created successfully in" in result.output

    # Verify the workflow directory was actually created
    workflow_dir = tmp_path / "my-workflow"
    assert workflow_dir.is_dir()

    # Cleanup created workflow
    if workflow_dir.exists():
        for child in workflow_dir.iterdir():
            child.unlink()
        workflow_dir.rmdir()
