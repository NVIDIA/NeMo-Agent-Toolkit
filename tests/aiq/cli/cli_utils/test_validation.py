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

import click
import pytest

from aiq.cli.cli_utils import validation


# Make a fixture which auto registers the test workflow
@pytest.fixture(autouse=True, scope="function")
def do_register_test_workflow(register_test_workflow):
    register_test_workflow()
    yield


@pytest.mark.usefixtures("register_test_workflow")
def test_validate_config(config_file: str):

    config_obj = validation.validate_config(config_file)
    assert config_obj.workflow.type == "test_workflow"


@pytest.mark.parametrize("config_file_name, expected_error_re",
                         [("invalid_yaml.yaml", r"^Invalid YAML format.*"),
                          ("missing_section_config.yaml", r"^Validation error: .*"),
                          ("missing_type_config.yaml", r"^Validation error: .*")])
def test_validate_config_error(test_data_dir: str, config_file_name: str, expected_error_re: str):
    config_file = os.path.join(test_data_dir, config_file_name)

    with pytest.raises(click.ClickException, match=expected_error_re):
        validation.validate_config(config_file)
