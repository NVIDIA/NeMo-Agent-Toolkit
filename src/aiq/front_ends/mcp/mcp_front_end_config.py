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

from pydantic import Field

from aiq.data_models.front_end import FrontEndBaseConfig


class MCPFrontEndConfig(FrontEndBaseConfig, name="mcp"):
    """MCP front end configuration.

    A simple MCP (Modular Communication Protocol) front end for AIQ.
    """

    name: str = Field(default="AIQ MCP", description="Name of the MCP server")
    host: str = Field(default="localhost", description="Host to bind the server to")
    port: int = Field(default=9901, description="Port to bind the server to", ge=0, le=65535)
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Log level for the MCP server")
    tool_names: list[str] = Field(default_factory=list, description="The list of tools MCP server will expose.")
