# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nat.cli.type_registry import RegisteredWorkspaceActionInfo
from nat.cli.type_registry import TypeRegistry
from nat.data_models.discovery_metadata import DiscoveryStatusEnum
from nat.workspace.types import TypeSchema
from nat.workspace.types import WorkspaceActionSchema


def test_register_function(registry: TypeRegistry):
    with pytest.raises(KeyError):
        registry.get_function(FunctionTestConfig)

    def tool_fn(builder: Builder):
        pass

    registry.register_function(
        RegisteredFunctionInfo(full_type="test/function", config_type=FunctionTestConfig, build_fn=tool_fn))

    workflow_info = registry.get_function(FunctionTestConfig)
    assert workflow_info.full_type == "test/function"
    assert workflow_info.module_name == "test"
    assert workflow_info.local_name == "function"
    assert workflow_info.config_type is FunctionTestConfig
    assert workflow_info.build_fn is tool_fn


def test_register_workspace_action(registry: TypeRegistry):
    schema = WorkspaceActionSchema(
        name="example",
        description="Example workspace action.",
        parameters=[TypeSchema(type="string", description="Example input.")],
        result=TypeSchema(type="object", description="Example output."),
    )
    action_cls = type("ExampleAction", (), {})

    registry.register_workspace_action(
        RegisteredWorkspaceActionInfo(name="example", action_cls=action_cls, action_schema=schema))

    action_info = registry.get_workspace_action("example")
    assert action_info.action_cls is action_cls
    assert action_info.action_schema == schema
    assert action_info.discovery_metadata.package == "test"
    assert action_info.discovery_metadata.version == "0.1.0"
    assert action_info.discovery_metadata.component_name == "example"
    assert action_info.discovery_metadata.description == "Example workspace action."
    assert action_info.discovery_metadata.developer_notes == ""
    assert action_info.discovery_metadata.status == DiscoveryStatusEnum.SUCCESS
