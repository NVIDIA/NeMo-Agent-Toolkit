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
from nat.plugins.a2a.server.front_end_plugin import A2AFrontEndPlugin
from nat.plugins.a2a.server.front_end_plugin_worker import A2AFrontEndPluginWorker


class TestA2AServerFunctionality:
    """Test A2A server functional behavior.

    These tests verify that the A2A server plugin correctly initializes,
    processes workflows, and creates the necessary components for serving
    NAT workflows as A2A agents.
    """

    async def test_server_plugin_initialization(self, a2a_server_config):
        """Test server plugin initializes correctly.

        Verifies that the A2A frontend plugin can be instantiated
        with proper configuration.
        """
        plugin = A2AFrontEndPlugin(full_config=a2a_server_config)

        assert plugin.front_end_config is not None
        assert plugin.front_end_config.name == "Test Agent"
        assert plugin.front_end_config.host == "localhost"
        assert plugin.front_end_config.port == 10000

    async def test_worker_extracts_all_functions(self, mock_workflow_with_functions, a2a_server_config):
        """Test worker extracts all functions from workflow.

        Verifies that the worker can discover and extract all functions
        from a workflow, which will be mapped to agent skills.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        functions = await worker._get_all_functions(mock_workflow_with_functions)

        # Verify all functions are extracted
        assert len(functions) == 3
        sep = FunctionGroup.SEPARATOR
        assert f"calculator{sep}add" in functions
        assert f"calculator{sep}multiply" in functions
        assert "current_datetime" in functions

        # Verify function objects are preserved
        assert functions[f"calculator{sep}add"].description == "Add two or more numbers together"

    async def test_agent_executor_creation(self, mock_workflow_with_functions, mock_workflow_builder,
                                           a2a_server_config):
        """Test agent executor is created correctly.

        Verifies that the worker creates a valid agent executor
        that can handle A2A protocol requests.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        executor = worker.create_agent_executor(mock_workflow_with_functions, mock_workflow_builder)

        # Verify executor is created
        assert executor is not None

        # Verify executor has required components
        assert hasattr(executor, 'session_manager')
        assert executor.session_manager is not None

    async def test_a2a_server_creation(self, mock_workflow_with_functions, mock_workflow_builder, a2a_server_config):
        """Test A2A server is created correctly.

        Verifies that the worker can create a complete A2A server
        with agent card and executor.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)

        # Create agent card and executor
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)
        executor = worker.create_agent_executor(mock_workflow_with_functions, mock_workflow_builder)

        # Create A2A server
        server = worker.create_a2a_server(agent_card, executor)

        # Verify server is created
        assert server is not None

        # Verify server has agent card
        assert hasattr(server, 'agent_card') or hasattr(server, '_agent_card')

    async def test_worker_config_access(self, a2a_server_config):
        """Test worker can access configuration correctly.

        Verifies that the worker properly stores and accesses
        the server configuration.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)

        # Verify worker has access to config
        assert worker.full_config is not None
        assert worker.full_config.general.front_end.name == "Test Agent"

    async def test_function_to_skill_transformation(self, mock_workflow_with_functions, a2a_server_config):
        """Test function to skill transformation logic.

        Verifies that the transformation from NAT functions to
        A2A skills preserves all necessary metadata.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        # Verify each function is properly transformed
        for skill in agent_card.skills:
            # Skill should have required fields
            assert skill.id is not None
            assert skill.name is not None
            assert skill.description is not None

            # Skill ID should match original function name
            assert skill.id in mock_workflow_with_functions.functions

    async def test_agent_protocol_version(self, mock_workflow_with_functions, a2a_server_config):
        """Test agent card includes correct protocol version.

        Verifies that the agent card specifies the A2A protocol
        version it implements.
        """
        worker = A2AFrontEndPluginWorker(a2a_server_config)
        agent_card = await worker.create_agent_card(mock_workflow_with_functions)

        # Verify protocol version is set
        assert hasattr(agent_card, 'protocol_version')
        assert agent_card.protocol_version is not None
        assert isinstance(agent_card.protocol_version, str)
