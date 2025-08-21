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

from click.testing import CliRunner

from nat.cli.commands.object_store.object_store import object_store_command


def test_object_store_command_help():
    """Test that the object-store command shows help correctly."""
    cli_runner = CliRunner()
    result = cli_runner.invoke(object_store_command, ["--help"])

    assert result.exit_code == 0
    assert "Manage object store operations" in result.output
    assert "upload" in result.output


def test_object_store_upload_command_help():
    """Test that the upload subcommand shows help correctly."""
    cli_runner = CliRunner()
    result = cli_runner.invoke(object_store_command, ["upload", "--help"])

    assert result.exit_code == 0
    assert "Upload a directory to an object store" in result.output
    assert "--store-type" in result.output
    assert "LOCAL_DIR" in result.output
    assert "--bucket-name" in result.output


def test_object_store_upload_command_missing_required_options():
    """Test that the upload command fails when required options are missing."""
    cli_runner = CliRunner()
    result = cli_runner.invoke(object_store_command, ["upload"])

    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_object_store_upload_command_invalid_store_type():
    """Test that the upload command fails with invalid store type."""
    cli_runner = CliRunner()
    result = cli_runner.invoke(object_store_command,
                               ["upload", "--store-type", "invalid", "/tmp", "--bucket-name", "test"])

    assert result.exit_code != 0
    assert "Invalid value for '--store-type'" in result.output
