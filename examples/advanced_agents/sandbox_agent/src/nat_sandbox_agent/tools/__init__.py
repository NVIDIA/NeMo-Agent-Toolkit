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

"""Tools module - Modular tool implementations.

Tool Architecture:
- Sandbox tools: Run inside Docker sandbox (shell, python, file_*, web_browse)
- Host tools: Run directly on host (web_search, youtube_transcript)
"""

from nat_sandbox_agent.tools.common import DEFAULT_MAX_OUTPUT_CHARS
from nat_sandbox_agent.tools.common import truncate_output
from nat_sandbox_agent.tools.factory import create_all_tools
from nat_sandbox_agent.tools.factory import get_tool_descriptions
from nat_sandbox_agent.tools.host import create_host_tools
from nat_sandbox_agent.tools.sandbox import SandboxToolExecutor
from nat_sandbox_agent.tools.sandbox import create_sandbox_tools

__all__ = [
    # Factory
    "create_all_tools",
    "get_tool_descriptions",
    # Sandbox
    "create_sandbox_tools",
    "SandboxToolExecutor",
    # Host
    "create_host_tools",
    # Utilities
    "DEFAULT_MAX_OUTPUT_CHARS",
    "truncate_output",
]
