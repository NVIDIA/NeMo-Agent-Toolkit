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
"""Test A2A client function group registration and behavior."""

from nat.builder.function import FunctionGroup


class TestA2AClientFunctionGroup:
    """Test A2A client function group registration and behavior."""

    async def test_all_api_levels_registered(self, a2a_function_group):
        """Test that all three API levels are registered.

        Verifies that the A2A client function group registers functions
        for all three API levels: high-level, helpers, and low-level.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # High-level function
        sep = FunctionGroup.SEPARATOR
        assert f"test_agent{sep}call" in functions, "High-level call function should be registered"

        # Helper functions
        assert f"test_agent{sep}get_skills" in functions, "get_skills helper should be registered"
        assert f"test_agent{sep}get_info" in functions, "get_info helper should be registered"
        assert f"test_agent{sep}get_task" in functions, "get_task helper should be registered"
        assert f"test_agent{sep}cancel_task" in functions, "cancel_task helper should be registered"

        # Low-level functions
        assert f"test_agent{sep}send_message" in functions, "send_message low-level function should be registered"
        assert f"test_agent{sep}send_message_streaming" in functions, "send_message_streaming should be registered"

        # Verify total count
        assert len(functions) == 7, "Should have exactly 7 functions registered"

    async def test_function_naming_conventions(self, a2a_function_group):
        """Test function names follow expected conventions.

        Verifies that all function names follow the pattern:
        {function_group_name}__{function_name}
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # All functions should start with the function group name
        sep = FunctionGroup.SEPARATOR
        prefix = f"test_agent{sep}"
        for func_name in functions.keys():
            assert func_name.startswith(prefix), \
                f"Function {func_name} should start with '{prefix}'"

        # Verify specific naming patterns
        expected_names = [
            f"test_agent{sep}call",
            f"test_agent{sep}get_skills",
            f"test_agent{sep}get_info",
            f"test_agent{sep}get_task",
            f"test_agent{sep}cancel_task",
            f"test_agent{sep}send_message",
            f"test_agent{sep}send_message_streaming",
        ]

        for expected in expected_names:
            assert expected in functions, f"Expected function {expected} not found"

    async def test_function_group_in_workflow(self, a2a_function_group):
        """Test function group works in workflow context.

        Verifies that the A2A client function group integrates
        correctly with the workflow builder.
        """
        group, _ = a2a_function_group

        # Verify the group has the correct config
        assert str(group._config.url) == "http://localhost:10000/"

        # Verify functions are accessible
        functions = await group.get_accessible_functions()
        assert len(functions) > 0

        # Verify each function can be retrieved and is callable
        for func_name, func in functions.items():
            assert func is not None
            assert hasattr(func, 'acall_invoke'), f"Function {func_name} should have acall_invoke method"

    async def test_function_signatures_correct(self, a2a_function_group):
        """Test function signatures match expected parameters.

        Verifies that each function has the correct input parameters
        and can be invoked with the expected arguments.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # Test high-level call function signature
        sep = FunctionGroup.SEPARATOR
        call_fn = functions[f"test_agent{sep}call"]
        assert call_fn.input_schema is not None

        # Verify call function accepts 'query' parameter
        schema_props = call_fn.input_schema.model_json_schema()["properties"]
        assert "query" in schema_props
        assert schema_props["query"]["type"] == "string"

        # Test send_message function signature
        send_msg_fn = functions[f"test_agent{sep}send_message"]
        schema_props = send_msg_fn.input_schema.model_json_schema()["properties"]
        assert "query" in schema_props
        # Optional parameters
        assert "task_id" in schema_props
        assert "context_id" in schema_props

    async def test_helper_functions_return_correct_types(self, a2a_function_group):
        """Test helper functions return expected data structures.

        Verifies that helper functions return data in the expected
        format with all required fields.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # Test get_skills return type
        sep = FunctionGroup.SEPARATOR
        get_skills_fn = functions[f"test_agent{sep}get_skills"]
        skills_result = await get_skills_fn.acall_invoke()

        assert isinstance(skills_result, dict)
        assert "agent" in skills_result
        assert "skills" in skills_result
        assert isinstance(skills_result["skills"], list)

        # Verify each skill has required fields
        for skill in skills_result["skills"]:
            assert "id" in skill
            assert "name" in skill
            assert "description" in skill
            assert "examples" in skill
            assert "tags" in skill

        # Test get_info return type
        get_info_fn = functions[f"test_agent{sep}get_info"]
        info_result = await get_info_fn.acall_invoke()

        assert isinstance(info_result, dict)
        assert "name" in info_result
        assert "version" in info_result
        assert "description" in info_result
        assert "url" in info_result
        assert "capabilities" in info_result
        assert "num_skills" in info_result
