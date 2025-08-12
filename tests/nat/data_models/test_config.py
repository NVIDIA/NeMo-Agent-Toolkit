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

import re
from io import StringIO

import pytest

from _utils.configs import WorkflowTestConfig
from nat.data_models.config import Config


# Make a fixture which auto registers the test workflow
@pytest.fixture(autouse=True, scope="function")
def do_register_test_workflow(register_test_workflow):
    register_test_workflow()
    yield


def test_aiq_config_print_summary(workflow_config: WorkflowTestConfig):

    c = Config(workflow=workflow_config)

    # We don't want to be strict about the exact format of the printed output, but we do want to assert that it printed
    # something relevant.
    workflow_name = workflow_config.type
    expected_re = re.compile(f"workflow.*:.*{workflow_name}", flags=(re.MULTILINE | re.IGNORECASE))

    buffer = StringIO()
    c.print_summary(stream=buffer)
    buffer.seek(0)
    assert expected_re.search(buffer.read()) is not None


def test_invalid_config_path():

    with pytest.raises(ValueError, match=re.compile(r"^functions\.invalid_function\.prompt$", re.MULTILINE)):
        Config.model_validate({
            'functions': {
                'invalid_function': {
                    '_type': 'test_workflow',
                    'llm_name': 'test',
                    'functions': ['test'],
                }
            }
        })
