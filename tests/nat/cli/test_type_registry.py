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

import pytest

from _utils.configs import FunctionTestConfig
from nat.builder.builder import Builder
from nat.cli.type_registry import RegisteredFunctionInfo
from nat.cli.type_registry import TypeRegistry


def test_register_function(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_function(FunctionTestConfig)

    def tool_fn(builder: Builder):  # pylint: disable=unused-argument
        pass

    registry.register_function(
        RegisteredFunctionInfo(full_type="test/function", config_type=FunctionTestConfig, build_fn=tool_fn))

    workflow_info = registry.get_function(FunctionTestConfig)
    assert workflow_info.full_type == "test/function"
    assert workflow_info.module_name == "test"
    assert workflow_info.local_name == "function"
    assert workflow_info.config_type is FunctionTestConfig
    assert workflow_info.build_fn is tool_fn
