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

from unittest.mock import MagicMock

import pytest

from nat.builder.function import FunctionGroup
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.plugins.a2a.server.front_end_config import A2AFrontEndConfig


@pytest.fixture(name="mock_workflow_builder")
def fixture_mock_workflow_builder() -> MagicMock:
    """Mock workflow builder for A2A server testing."""
    return MagicMock()


@pytest.fixture(name="mock_workflow_with_functions")
def fixture_mock_workflow_with_functions() -> MagicMock:
    """Mock workflow with test functions for A2A server testing."""
    mock_workflow = MagicMock()

    # Create mock functions with realistic attributes
    sep = FunctionGroup.SEPARATOR
    add_fn = MagicMock()
    add_fn.name = f"calculator{sep}add"
    add_fn.description = "Add two or more numbers together"
    add_fn.input_schema = {"type": "object", "properties": {"numbers": {"type": "array"}}}

    multiply_fn = MagicMock()
    multiply_fn.name = f"calculator{sep}multiply"
    multiply_fn.description = "Multiply two or more numbers together"
    multiply_fn.input_schema = {"type": "object", "properties": {"numbers": {"type": "array"}}}

    datetime_fn = MagicMock()
    datetime_fn.name = "current_datetime"
    datetime_fn.description = "Get current date and time"
    datetime_fn.input_schema = {"type": "object", "properties": {}}

    mock_workflow.functions = {
        f"calculator{sep}add": add_fn, f"calculator{sep}multiply": multiply_fn, "current_datetime": datetime_fn
    }
    mock_workflow.function_groups = {}

    return mock_workflow


@pytest.fixture(name="a2a_server_config")
def fixture_a2a_server_config() -> Config:
    """Sample A2A server configuration for testing."""
    return Config(general=GeneralConfig(front_end=A2AFrontEndConfig(
        name="Test Agent", description="Test agent for unit tests", host="localhost", port=10000, version="1.0.0")))
