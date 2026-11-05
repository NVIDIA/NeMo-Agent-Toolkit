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
"""A function a group makes available as a tool can be the entry point of a workflow (as MCP servers do)."""

import pytest

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.cli.register_workflow import register_function_group
from nat.cli.register_workflow import register_per_user_function
from nat.cli.register_workflow import register_per_user_function_group
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.function import FunctionGroupBaseConfig
from nat.runtime.session import SessionManager


class TextToolsConfig(FunctionGroupBaseConfig, name="entry_test_text_tools"):
    pass


class EchoWorkflowConfig(FunctionBaseConfig, name="entry_test_echo_workflow"):
    pass


class PerUserTextToolsConfig(FunctionGroupBaseConfig, name="entry_test_per_user_text_tools"):
    pass


class PerUserEchoWorkflowConfig(FunctionBaseConfig, name="entry_test_per_user_echo_workflow"):
    pass


def _text_tools_group(config) -> FunctionGroup:
    group = FunctionGroup(config=config)

    async def upper(text: str) -> str:
        return text.upper()

    async def reverse(text: str) -> str:
        return text[::-1]

    group.add_function("upper", upper, description="upper-case the text")
    group.add_function("reverse", reverse, description="reverse the text")
    return group


@pytest.fixture(name="register_components", scope="module", autouse=True)
def register_components_fixture():

    @register_function_group(config_type=TextToolsConfig)
    async def _text_tools(config: TextToolsConfig, builder: Builder):
        yield _text_tools_group(config)

    @register_per_user_function_group(config_type=PerUserTextToolsConfig)
    async def _per_user_text_tools(config: PerUserTextToolsConfig, builder: Builder):
        yield _text_tools_group(config)

    @register_per_user_function(config_type=PerUserEchoWorkflowConfig, input_type=str, single_output_type=str)
    async def _per_user_workflow(config, builder: Builder):

        async def _inner(text: str) -> str:
            return text

        yield FunctionInfo.from_fn(_inner, description="echo")

    @register_function(config_type=EchoWorkflowConfig)
    async def _workflow(config, builder: Builder):

        async def _inner(text: str) -> str:
            return text

        yield FunctionInfo.from_fn(_inner, description="echo")


def _config(group_type: str = "entry_test_text_tools",
            workflow_type: str = "entry_test_echo_workflow",
            **group_options) -> Config:
    return Config.model_validate({
        "function_groups": {
            "text": {
                "_type": group_type, **group_options
            }
        },
        "workflow": {
            "_type": workflow_type
        },
    })


async def _run(config: Config, entry_function: str, user_id: str | None = None) -> str:
    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(config=config,
                                                      shared_builder=builder,
                                                      entry_function=entry_function)
        try:
            async with session_manager.session(user_id=user_id) as session:
                async with session.run("abc") as runner:
                    return await runner.result(to_type=str)
        finally:
            await session_manager.shutdown()


@pytest.mark.parametrize("group_options", [{}, {
    "exclude": ["reverse"]
}, {
    "include": ["upper"]
}],
                         ids=["no_filter", "exclude", "include"])
async def test_an_available_group_function_can_be_the_entry_function(group_options):
    assert await _run(_config(**group_options), "text__upper") == "ABC"


@pytest.mark.parametrize("entry_function", ["text__reverse", "text__missing", "missing"])
async def test_a_function_that_is_not_available_is_still_not_found(entry_function):
    with pytest.raises(ValueError, match="not found"):
        await _run(_config(exclude=["reverse"]), entry_function)


@pytest.mark.parametrize("group_type", ["entry_test_text_tools", "entry_test_per_user_text_tools"],
                         ids=["shared_group", "per_user_group"])
async def test_a_per_user_workflow_can_use_a_group_function_as_the_entry_function(group_type):
    config = _config(group_type=group_type, workflow_type="entry_test_per_user_echo_workflow")
    assert await _run(config, "text__upper", user_id="user-1") == "ABC"


class NestedToolsConfig(FunctionGroupBaseConfig, name="entry_test_nested_tools"):
    function_name: str


@pytest.fixture(name="register_nested_group", scope="module")
def register_nested_group_fixture():

    @register_function_group(config_type=NestedToolsConfig)
    async def _nested_tools(config: NestedToolsConfig, builder: Builder):
        group = FunctionGroup(config=config)

        async def nested(text: str) -> str:
            return text

        group.add_function(config.function_name, nested, description="a function whose name may hold the separator")
        yield group


@pytest.mark.usefixtures("register_nested_group")
async def test_two_groups_making_the_same_name_available_is_an_error():
    """Group `a` with `b__c` and group `a__b` with `c` both offer `a__b__c`."""
    config = Config.model_validate({
        "function_groups": {
            "a": {
                "_type": "entry_test_nested_tools", "function_name": "b__c"
            },
            "a__b": {
                "_type": "entry_test_nested_tools", "function_name": "c"
            }
        },
        "workflow": {
            "_type": "entry_test_echo_workflow"
        },
    })
    with pytest.raises(ValueError, match="ambiguous"):
        await _run(config, "a__b__c")
