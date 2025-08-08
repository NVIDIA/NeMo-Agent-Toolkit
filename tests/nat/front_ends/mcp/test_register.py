# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.data_models.config import Config
from aiq.data_models.config import GeneralConfig
from aiq.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig
from aiq.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin
from aiq.front_ends.mcp.register import register_mcp_front_end
from aiq.test.functions import EchoFunctionConfig


async def test_register_mcp_front_end():
    """Test that the register_mcp_front_end function returns the correct plugin."""
    # Create configuration objects
    mcp_config = MCPFrontEndConfig(name="Test MCP Server")

    # Use a real AIQConfig with a proper workflow
    full_config = Config(general=GeneralConfig(front_end=mcp_config), workflow=EchoFunctionConfig())

    # Use the context manager pattern since register_mcp_front_end
    # returns an AsyncGeneratorContextManager, not an async iterator
    async with register_mcp_front_end(mcp_config, full_config) as plugin:
        # Verify that the plugin is of the correct type and has the right config
        assert isinstance(plugin, MCPFrontEndPlugin)
        assert plugin.full_config is full_config
