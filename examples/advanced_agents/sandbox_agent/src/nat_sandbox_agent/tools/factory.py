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

"""Factory for creating all agent tools (sandbox + host)."""


from langchain_core.tools import StructuredTool

from nat_sandbox_agent.sandbox.base import BaseSandbox
from nat_sandbox_agent.tools.common import DEFAULT_MAX_OUTPUT_CHARS
from nat_sandbox_agent.tools.host import create_host_tools
from nat_sandbox_agent.tools.sandbox import create_sandbox_tools


def create_all_tools(
    sandbox: BaseSandbox,
    tavily_api_key: str | None = None,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    include_tools: list[str] | None = None,
) -> list[StructuredTool]:
    """Create all tools (sandbox + host).

    This function combines:
    - Sandbox tools: shell, python, file_read, file_write, web_browse
    - Host tools: web_search, youtube_transcript

    Args:
        sandbox: Sandbox instance for sandbox tools.
        tavily_api_key: API key for web search. If None, uses env var.
        max_output_chars: Maximum characters for tool output truncation.
        include_tools: Optional list of tool names to include.
            If None, all tools are included.

    Returns:
        Combined list of all tools.
    """
    # Sandbox tools (shell, python, file_read, file_write, web_browse)
    sandbox_tools = create_sandbox_tools(
        sandbox=sandbox,
        max_output_chars=max_output_chars,
    )

    # Host tools (web_search, youtube_transcript)
    host_tools = create_host_tools(
        tavily_api_key=tavily_api_key,
        max_output_chars=max_output_chars,
    )

    all_tools = {t.name: t for t in sandbox_tools + host_tools}

    if include_tools:
        return [all_tools[name] for name in include_tools if name in all_tools]

    return list(all_tools.values())

def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools.

    Returns:
        Formatted string with tool names and descriptions.
    """
    tools_info = [
        # Sandbox tools
        ("shell", "Execute bash commands for system operations"),
        ("python", "Execute Python code for data processing and analysis"),
        ("file_read", "Read file contents from the sandbox"),
        ("file_write", "Write content to a file in the sandbox"),
        ("web_browse", "Browse webpages and extract content"),
        # Host tools
        ("web_search", "Search the web using Tavily"),
        ("youtube_transcript", "Get transcript from YouTube videos"),
    ]

    lines = ["Available tools:"]
    for name, desc in tools_info:
        lines.append(f"  - {name}: {desc}")

    return "\n".join(lines)
