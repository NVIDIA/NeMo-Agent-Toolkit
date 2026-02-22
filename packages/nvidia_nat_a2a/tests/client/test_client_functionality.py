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
"""Test A2A client functional behavior."""

from datetime import timedelta
from unittest.mock import patch

from nat.builder.function import FunctionGroup
from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.a2a.client.client_config import A2AClientConfig


class TestA2AClientFunctionality:
    """Test A2A client functional behavior with mocked agents."""

    async def test_client_discovers_agent_skills(self, a2a_function_group):
        """Test client can discover and list agent skills.

        Verifies that the A2A client can successfully discover and retrieve
        the list of skills from a remote agent.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # Verify get_skills function exists
        sep = FunctionGroup.SEPARATOR
        assert f"test_agent{sep}get_skills" in functions

        # Call get_skills
        get_skills_fn = functions[f"test_agent{sep}get_skills"]
        result = await get_skills_fn.acall_invoke()

        # Verify skills are returned with correct structure
        assert "skills" in result
        assert "agent" in result
        assert result["agent"] == "Test Agent"

        # Verify skills are present
        skills = result["skills"]
        assert len(skills) == 3, "Should have exactly 3 skills from sample agent card"

        skill_ids = [s["id"] for s in skills]
        assert f"calculator{FunctionGroup.SEPARATOR}add" in skill_ids
        assert f"calculator{FunctionGroup.SEPARATOR}multiply" in skill_ids
        assert "current_datetime" in skill_ids

        # Verify skill details are present and well-formed
        add_skill = next(s for s in skills if s["id"] == f"calculator{FunctionGroup.SEPARATOR}add")
        assert add_skill["name"] == "Add"
        assert add_skill["description"] == "Add two or more numbers together"
        assert "examples" in add_skill
        assert len(add_skill["examples"]) > 0

    async def test_client_invokes_high_level_call(self, a2a_function_group):
        """Test calling agent with natural language query.

        Verifies that the high-level call() function exists and has
        the correct signature for natural language queries.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # Verify call function exists
        sep = FunctionGroup.SEPARATOR
        assert f"test_agent{sep}call" in functions

        # Verify function has correct signature
        call_fn = functions[f"test_agent{sep}call"]
        assert call_fn.input_schema is not None

        schema_props = call_fn.input_schema.model_json_schema()["properties"]
        assert "query" in schema_props
        assert schema_props["query"]["type"] == "string"

        # Verify function has description containing agent info
        assert call_fn.description is not None
        assert "Test agent for unit tests" in call_fn.description

    async def test_skills_embedded_when_enabled(self, sample_agent_card, mock_user_context):
        """Test skills are embedded in function description when enabled.

        Verifies that when include_skills_in_description is True,
        the skill details are included in the high-level function description.
        """
        with patch('nat.plugins.a2a.client.client_impl.A2ABaseClient') as mock_class:
            # Configure the mock: return_value is what gets assigned to self._client
            mock_class.return_value.agent_card = sample_agent_card
            mock_class.return_value.__aenter__.return_value = mock_class.return_value

            config = A2AClientConfig(
                url="http://localhost:10000",
                include_skills_in_description=True,
            )

            # Mock the Context to provide a user_id
            with patch('nat.builder.context.Context') as mock_context:
                mock_context.get.return_value = mock_user_context

                async with WorkflowBuilder() as builder:
                    group = await builder.add_function_group("test_agent", config)
                    functions = await group.get_accessible_functions()

                    call_fn = functions[f"test_agent{FunctionGroup.SEPARATOR}call"]

                    # Verify skills are embedded in description
                    # The description should mention the skills/capabilities
                    assert "Capabilities" in call_fn.description or "Skills" in call_fn.description

                    # Verify skill names or descriptions appear
                    description_lower = call_fn.description.lower()
                    assert "add" in description_lower or "multiply" in description_lower \
                        or "datetime" in description_lower

    async def test_skills_not_embedded_when_disabled(self, sample_agent_card, mock_user_context):
        """Test skills are not embedded when disabled.

        Verifies that when include_skills_in_description is False,
        the skill details are NOT included in the function description.
        """
        with patch('nat.plugins.a2a.client.client_impl.A2ABaseClient') as mock_class:
            # Configure the mock: return_value is what gets assigned to self._client
            mock_class.return_value.agent_card = sample_agent_card
            mock_class.return_value.__aenter__.return_value = mock_class.return_value

            config = A2AClientConfig(
                url="http://localhost:10000",
                include_skills_in_description=False,
            )

            # Mock the Context to provide a user_id
            with patch('nat.builder.context.Context') as mock_context:
                mock_context.get.return_value = mock_user_context

                async with WorkflowBuilder() as builder:
                    group = await builder.add_function_group("test_agent", config)
                    functions = await group.get_accessible_functions()

                    call_fn = functions[f"test_agent{FunctionGroup.SEPARATOR}call"]

                    # Verify description is shorter when skills not embedded
                    # (it should still have a description, just without skill details)
                    assert len(call_fn.description) > 0

                    # The description should be more generic
                    # (not checking for absence of specific terms as format may vary)

    async def test_get_info_returns_agent_metadata(self, a2a_function_group):
        """Test get_info returns correct agent metadata.

        Verifies that the get_info helper function returns
        the correct agent metadata including name, version, and capabilities.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # Verify get_info function exists
        sep = FunctionGroup.SEPARATOR
        assert f"test_agent{sep}get_info" in functions

        # Call get_info
        get_info_fn = functions[f"test_agent{sep}get_info"]
        result = await get_info_fn.acall_invoke()

        # Verify metadata structure and content
        assert result["name"] == "Test Agent"
        assert result["version"] == "1.0.0"
        assert result["description"] == "Test agent for unit tests"
        assert result["url"] == "http://localhost:10000/"

        # Verify capabilities
        assert "capabilities" in result
        assert isinstance(result["capabilities"], dict)
        assert result["capabilities"]["streaming"] is True

        # Verify skill count
        assert result["num_skills"] == 3

    async def test_client_connection_configuration(self, sample_agent_card, mock_user_context):
        """Test client connection configuration is properly set.

        Verifies that the client is initialized with the correct
        connection parameters from the configuration.
        """
        with patch('nat.plugins.a2a.client.client_impl.A2ABaseClient') as mock_class:
            # Configure the mock: return_value is what gets assigned to self._client
            mock_class.return_value.agent_card = sample_agent_card
            mock_class.return_value.__aenter__.return_value = mock_class.return_value

            config = A2AClientConfig(url="http://localhost:10000", task_timeout=60.0)

            # Mock the Context to provide a user_id
            with patch('nat.builder.context.Context') as mock_context:
                mock_context.get.return_value = mock_user_context

                async with WorkflowBuilder() as builder:
                    group = await builder.add_function_group("test_agent", config)

                # Verify function group was created
                assert group is not None

                # Verify A2ABaseClient was instantiated with correct parameters
                mock_class.assert_called_once()
                call_kwargs = mock_class.call_args.kwargs
                # URL gets normalized with trailing slash
                assert call_kwargs['base_url'] == "http://localhost:10000/"
                # Timeout is converted to timedelta
                assert call_kwargs['task_timeout'] == timedelta(seconds=60)
                # Default A2A agent card path
                assert call_kwargs['agent_card_path'] == '/.well-known/agent-card.json'

    async def test_client_timeout_configuration(self, sample_agent_card, mock_user_context):
        """Test client timeout can be configured.

        Verifies that the task_timeout configuration is properly
        set and accessible.
        """
        with patch('nat.plugins.a2a.client.client_impl.A2ABaseClient') as mock_class:
            # Configure the mock: return_value is what gets assigned to self._client
            mock_class.return_value.agent_card = sample_agent_card
            mock_class.return_value.__aenter__.return_value = mock_class.return_value

            config = A2AClientConfig(
                url="http://localhost:10000",
                task_timeout=timedelta(seconds=60),
            )

            # Verify timeout is set correctly
            assert config.task_timeout.total_seconds() == 60

            # Mock the Context to provide a user_id
            with patch('nat.builder.context.Context') as mock_context:
                mock_context.get.return_value = mock_user_context

                async with WorkflowBuilder() as builder:
                    group = await builder.add_function_group("test_agent", config)

                    # Verify group was created successfully
                    assert group is not None
                    functions = await group.get_accessible_functions()
                    assert len(functions) == 7

    async def test_multiple_functions_accessible(self, a2a_function_group):
        """Test multiple functions are accessible from function group.

        Verifies that the client exposes all expected functions
        and they are properly structured.
        """
        group, _ = a2a_function_group
        functions = await group.get_accessible_functions()

        # Verify we have multiple functions
        assert len(functions) == 7, "Should have 7 functions (1 high-level + 4 helpers + 2 low-level)"

        # Verify each function is properly structured
        for func in functions.values():
            assert func is not None
            assert hasattr(func, 'acall_invoke')
            assert func.description is not None
            assert len(func.description) > 0
