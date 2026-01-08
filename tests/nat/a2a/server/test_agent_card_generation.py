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

from nat.builder.function import FunctionGroup
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker


class TestAgentCardGeneration:
    """Test agent card creation from workflows.

    These tests verify that the A2A server correctly generates agent cards
    from NAT workflows, mapping functions to skills with proper metadata.
    """

    async def test_agent_card_includes_all_functions(self, mock_workflow_with_functions, a2a_server_config):
        """Test agent card includes all workflow functions as skills.

        Verifies that every function in the workflow is represented
        as a skill in the generated agent card.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        # Verify all functions are mapped to skills
        assert len(agent_card.skills) == 3

        skill_ids = [skill.id for skill in agent_card.skills]
        sep = FunctionGroup.SEPARATOR
        assert f"calculator{sep}add" in skill_ids
        assert f"calculator{sep}multiply" in skill_ids
        assert "current_datetime" in skill_ids

    async def test_skill_names_formatted_correctly(self, mock_workflow_with_functions, a2a_server_config):
        """Test skill names are formatted from function names.

        Verifies the transformation: "calculator__add" -> "Calculator - Add"
        This makes skill names more human-readable in the agent card.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        # Find the calculator__add skill
        sep = FunctionGroup.SEPARATOR
        add_skill = next(s for s in agent_card.skills if s.id == f"calculator{sep}add")

        # Verify name transformation: calculator__add -> Calculator - Add
        assert add_skill.name == "Calculator - Add"

        # Find the current_datetime skill
        datetime_skill = next(s for s in agent_card.skills if s.id == "current_datetime")

        # Verify name transformation: current_datetime -> Current Datetime
        assert datetime_skill.name == "Current Datetime"

    async def test_skill_descriptions_from_functions(self, mock_workflow_with_functions, a2a_server_config):
        """Test skill descriptions come from function descriptions.

        Verifies that function descriptions are preserved in the
        agent card skills.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        sep = FunctionGroup.SEPARATOR
        add_skill = next(s for s in agent_card.skills if s.id == f"calculator{sep}add")
        assert add_skill.description == "Add two or more numbers together"

        multiply_skill = next(s for s in agent_card.skills if s.id == f"calculator{sep}multiply")
        assert multiply_skill.description == "Multiply two or more numbers together"

    async def test_agent_card_metadata_from_config(self, mock_workflow_with_functions, a2a_server_config):
        """Test agent card metadata comes from configuration.

        Verifies that agent-level metadata (name, version, description)
        is correctly populated from the server configuration.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        assert agent_card.name == "Test Agent"
        assert agent_card.version == "1.0.0"
        assert agent_card.description == "Test agent for unit tests"

    async def test_agent_card_url_generation(self, mock_workflow_with_functions, a2a_server_config):
        """Test agent card URL is generated correctly.

        Verifies that the agent URL is constructed from host and port
        configuration with proper formatting.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        # URL should be formatted as http://host:port/
        assert agent_card.url == "http://localhost:10000/"

    async def test_agent_card_capabilities_from_config(self, mock_workflow_with_functions, a2a_server_config):
        """Test agent card capabilities from configuration.

        Verifies that agent capabilities (streaming, push notifications)
        are correctly set from configuration.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        # Verify capabilities structure exists
        assert agent_card.capabilities is not None
        assert hasattr(agent_card.capabilities, 'streaming')
        assert hasattr(agent_card.capabilities, 'push_notifications')

    async def test_empty_workflow_creates_valid_card(self, a2a_server_config):
        """Test agent card creation with empty workflow.

        Verifies that the server can create a valid agent card
        even when the workflow has no functions.
        """
        # Create workflow with no functions
        empty_workflow = type('MockWorkflow', (), {'functions': {}, 'function_groups': {}})()

        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(empty_workflow)

        # Agent card should still be valid with metadata
        assert agent_card.name == "Test Agent"
        assert agent_card.version == "1.0.0"
        assert len(agent_card.skills) == 0
