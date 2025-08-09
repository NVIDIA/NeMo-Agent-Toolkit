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

from collections.abc import AsyncIterator

from aiq.cli.register_workflow import register_front_end
from aiq.data_models.config import Config
from aiq.front_ends.mcp.mcp_front_end_config import MCPFrontEndConfig


@register_front_end(config_type=MCPFrontEndConfig)
async def register_mcp_front_end(config: MCPFrontEndConfig, full_config: Config) -> AsyncIterator:
    from aiq.front_ends.mcp.mcp_front_end_plugin import MCPFrontEndPlugin

    yield MCPFrontEndPlugin(full_config=full_config)
