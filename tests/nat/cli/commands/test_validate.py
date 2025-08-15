# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import re

import pytest
from click.testing import CliRunner

from nat.cli.commands.validate import validate_command


# Make a fixture which auto registers the test workflow
@pytest.fixture(autouse=True, scope="function")
def do_register_test_workflow(register_test_workflow):
    register_test_workflow()
    yield


@pytest.mark.parametrize("config_file_name, expected_pat, expected_exit_code",
                         [("config.yaml", r"configuration file is valid", 0),
                          ("invalid_yaml.yaml", r"validation failed", 1),
                          ("missing_section_config.yaml", r"validation failed", 1),
                          ("missing_type_config.yaml", r"validation failed", 1)])
def test_validate_command(test_data_dir: str, config_file_name: str, expected_pat: str, expected_exit_code: int):
    expected_re = re.compile(expected_pat, flags=(re.MULTILINE | re.IGNORECASE))
    config_file = os.path.join(test_data_dir, config_file_name)

    cli_runner = CliRunner()
    result = cli_runner.invoke(validate_command, ["--config_file", config_file])
    assert result.exit_code == expected_exit_code
    assert expected_re.search(result.output) is not None
