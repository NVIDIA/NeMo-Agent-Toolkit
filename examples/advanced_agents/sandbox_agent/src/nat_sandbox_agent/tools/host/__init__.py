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

"""Host-side tools that run directly on the host machine.

These tools don't require sandbox isolation and run directly on the host:
- web_search: Tavily API calls
- youtube_transcript: YouTube transcript API calls

This provides better security (API keys not exposed to sandbox) and
lower latency (no Docker exec overhead).
"""


from langchain_core.tools import StructuredTool

from nat_sandbox_agent.tools.common import DEFAULT_MAX_OUTPUT_CHARS
from nat_sandbox_agent.tools.host.web_search import create_web_search_tool
from nat_sandbox_agent.tools.host.youtube import create_youtube_tool


def create_host_tools(
    tavily_api_key: str | None = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> list[StructuredTool]:
    """Create all host-side tools.

    Args:
        tavily_api_key: Tavily API key for web search. If None, uses env var.
        max_output_chars: Maximum characters for tool output.

    Returns:
        List of host-side tools.
    """
    return [
        create_web_search_tool(api_key=tavily_api_key),
        create_youtube_tool(max_output_chars=max_output_chars),
    ]

__all__ = [
    "create_host_tools",
    "create_web_search_tool",
    "create_youtube_tool",
]
