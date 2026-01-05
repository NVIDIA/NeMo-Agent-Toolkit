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
"""
Banking Tools Registration for Agent Leaderboard Evaluation.

This module registers all banking tools from the Agent Leaderboard v2 dataset
as stubs that capture intent without execution using a function group.
"""

import json
import logging
from pathlib import Path

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

from .tool_intent_stubs import ToolIntentBuffer
from .tool_intent_stubs import create_tool_stub_function

logger = logging.getLogger(__name__)


class BankingToolsGroupConfig(FunctionGroupBaseConfig, name="banking_tools_group"):
    """
    Configuration for loading banking tools as a function group.

    This registers all banking tools from the raw dataset tools.json
    as stubs that capture intent without execution. Each tool is accessible
    individually by name (e.g., get_account_balance).
    """

    tools_json_path: str = Field(
        default="data/raw/banking/tools.json",
        description="Path to tools.json file containing banking tool schemas",
    )
    decision_only: bool = Field(
        default=True,
        description="If True, register tools as stubs. If False, skip registration.",
    )


@register_function_group(config_type=BankingToolsGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def banking_tools_group_function(config: BankingToolsGroupConfig, builder: Builder):
    """
    Registers all banking tools from tools.json as a function group.

    This creates a function group where each banking tool is an intent-capturing stub.
    Tools are globally accessible by name for use in workflow tool_names lists.

    Args:
        config: Configuration for banking tools group
        builder: NAT builder object

    Returns:
        FunctionGroup with all banking tool stubs
    """
    # Create the function group
    group = FunctionGroup(config=config)

    # Only register tools if decision_only is enabled
    if not config.decision_only:
        logger.info("decision_only is False, skipping banking tools stub registration")
        yield group  # Yield empty group
        return

    # Get or create intent buffer
    if not hasattr(builder, "runtime_metadata"):
        builder.runtime_metadata = {}

    intent_buffer = builder.runtime_metadata.get("tool_intent_buffer")
    if intent_buffer is None:
        intent_buffer = ToolIntentBuffer()
        builder.runtime_metadata["tool_intent_buffer"] = intent_buffer
        logger.info("Created new ToolIntentBuffer")

    # Load tools.json
    tools_path = Path(__file__).parent / config.tools_json_path
    if not tools_path.exists():
        # Try absolute path
        tools_path = Path(config.tools_json_path)

    if not tools_path.exists():
        logger.error("tools.json not found at %s", tools_path)
        raise FileNotFoundError(f"Banking tools file not found: {tools_path}")

    with open(tools_path) as f:
        tools_schemas = json.load(f)

    logger.info("Loaded %d tool schemas from %s", len(tools_schemas), tools_path)

    # Add each tool as a stub to the group
    registered_count = 0
    tool_names = []
    failed_tools = []

    for tool_schema in tools_schemas:
        tool_name = tool_schema.get("title", "")
        if not tool_name:
            logger.warning("Skipping tool with no title: %s", tool_schema)
            continue

        try:
            # Create stub function with custom input schema
            stub_fn, custom_input_schema, description = create_tool_stub_function(tool_schema, intent_buffer)

            # Add function to the group with custom input schema
            group.add_function(
                name=tool_name,
                fn=stub_fn,
                input_schema=custom_input_schema,
                description=description,
            )

            tool_names.append(tool_name)
            registered_count += 1
            logger.info("✓ Added banking tool stub: %s", tool_name)

        except Exception as e:
            logger.error("✗ Failed to add tool stub for %s: %s", tool_name, e, exc_info=True)
            failed_tools.append(tool_name)
            continue

    logger.info("Successfully registered %d/%d banking tool stubs in function group",
                registered_count,
                len(tools_schemas))
    if tool_names:
        logger.info(
            "Registered tools: %s",
            ", ".join(tool_names[:5]) +
            f"... and {len(tool_names)-5} more" if len(tool_names) > 5 else ", ".join(tool_names))
    if failed_tools:
        logger.warning("Failed to register %d tools: %s", len(failed_tools), ", ".join(failed_tools))

    # Yield the function group
    yield group
