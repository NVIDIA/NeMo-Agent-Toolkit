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

from unittest.mock import MagicMock
from unittest.mock import patch

from nat.cli.commands.finetune import finetune_command
from nat.data_models.config import Config
from nat.data_models.finetuning import FinetuneRunConfig


@patch("nat.cli.commands.finetune.discover_and_register_plugins")
@patch("nat.cli.commands.finetune.run_finetuning_sync")
@patch("nat.cli.commands.finetune.load_and_override_config")
@patch("nat.cli.commands.finetune.validate_schema")
def test_finetune_command_with_overrides(mock_validate, mock_load, mock_run, mock_discover, tmp_path):
    """Test that finetune_command correctly applies overrides and passes a Config object via Click CLI."""
    from click.testing import CliRunner

    mock_config_dict = {"finetuning": {"enabled": True}}
    mock_load.return_value = mock_config_dict

    mock_config = MagicMock(spec=Config)
    mock_validate.return_value = mock_config

    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("dummy")
    resolved_config_path = config_path.resolve()

    runner = CliRunner()
    result = runner.invoke(
        finetune_command,
        ["--config_file", str(config_path), "-o", "finetuning.num_epochs", "10", "--result_json_path", "$"],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"

    # Ensure plugins were discovered
    mock_discover.assert_called_once()

    # Ensure config was loaded and overridden correctly
    expected_overrides = (("finetuning.num_epochs", "10"), )
    mock_load.assert_called_once_with(resolved_config_path, expected_overrides)
    mock_validate.assert_called_once_with(mock_config_dict, Config)

    # Check that run_finetuning_sync was called with FinetuneRunConfig
    mock_run.assert_called_once()
    run_config_arg = mock_run.call_args[0][0]
    assert isinstance(run_config_arg, FinetuneRunConfig)
    assert run_config_arg.config_file is mock_config
    assert run_config_arg.override == expected_overrides


@patch("nat.cli.commands.finetune.discover_and_register_plugins")
@patch("nat.cli.commands.finetune.run_finetuning_sync")
@patch("nat.cli.commands.finetune.load_and_override_config")
@patch("nat.cli.commands.finetune.validate_schema")
def test_finetune_command_without_overrides(mock_validate, mock_load, mock_run, mock_discover, tmp_path):
    """Test the happy path when no overrides are provided via Click CLI."""
    from click.testing import CliRunner

    mock_config_dict = {"finetuning": {"enabled": True}}
    mock_load.return_value = mock_config_dict

    mock_config = MagicMock(spec=Config)
    mock_validate.return_value = mock_config

    config_path = tmp_path / "dummy.yaml"
    config_path.write_text("dummy")
    resolved_config_path = config_path.resolve()

    runner = CliRunner()
    result = runner.invoke(finetune_command, ["--config_file", str(config_path), "--result_json_path", "$"])

    assert result.exit_code == 0, f"Command failed: {result.output}"

    mock_discover.assert_called_once()
    mock_load.assert_called_once_with(resolved_config_path, ())
    mock_validate.assert_called_once_with(mock_config_dict, Config)

    mock_run.assert_called_once()
    run_config_arg = mock_run.call_args[0][0]
    assert isinstance(run_config_arg, FinetuneRunConfig)
    assert run_config_arg.config_file is mock_config
    assert run_config_arg.override == ()
