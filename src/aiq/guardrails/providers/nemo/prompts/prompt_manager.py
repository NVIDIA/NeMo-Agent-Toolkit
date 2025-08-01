# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt manager for automatic guardrails configuration."""

import logging
from typing import Any

from .defaults import DEFAULT_BOT_MESSAGES
from .defaults import DEFAULT_COLANG_FLOWS
from .defaults import DEFAULT_PROMPTS

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages default prompts and automatic guardrails configuration."""

    def __init__(self):
        self.prompts = DEFAULT_PROMPTS.copy()
        self.colang_flows = DEFAULT_COLANG_FLOWS.copy()
        self.bot_messages = DEFAULT_BOT_MESSAGES.copy()

    def get_prompt(self, prompt_name: str) -> str | None:
        """Get a default prompt by name."""
        return self.prompts.get(prompt_name)

    def get_colang_flow(self, flow_name: str) -> str | None:
        """Get a default Colang flow by name."""
        return self.colang_flows.get(flow_name)

    def get_bot_message(self, message_name: str) -> str | None:
        """Get a default bot message by name."""
        return self.bot_messages.get(message_name)

    def add_custom_prompt(self, name: str, prompt: str) -> None:
        """Add a custom prompt to the manager."""
        self.prompts[name] = prompt

    def add_custom_flow(self, name: str, flow: str) -> None:
        """Add a custom Colang flow to the manager."""
        self.colang_flows[name] = flow

    def add_custom_bot_message(self, name: str, message: str) -> None:
        """Add a custom bot message to the manager."""
        self.bot_messages[name] = message

    def generate_nemo_guardrails_config(
        self,
        input_flows: list[str] | None = None,
        output_flows: list[str] | None = None,
        custom_prompts: dict[str, str] | None = None,
        custom_flows: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a complete NeMo Guardrails configuration with default prompts.

        Args:
            input_flows: List of input flow names to include
            output_flows: List of output flow names to include
            custom_prompts: Custom prompts to override defaults
            custom_flows: Custom Colang flows to override defaults

        Returns:
            Complete NeMo Guardrails configuration dictionary
        """
        input_flows = input_flows or []
        output_flows = output_flows or []
        custom_prompts = custom_prompts or {}
        custom_flows = custom_flows or {}

        # Start with defaults and add customs
        all_prompts = {**self.prompts, **custom_prompts}
        all_flows = {**self.colang_flows, **custom_flows}

        # Generate YAML configuration
        yaml_config = {
            "models": [],  # Will be populated by the GuardrailsManager
            "rails": {
                "input": {
                    "flows": input_flows
                },
                "output": {
                    "flows": output_flows
                },
            },
        }

        # Generate prompts section if we have flows that need them
        all_flow_names = set(input_flows + output_flows)
        needed_prompts = {}

        for flow_name in all_flow_names:
            # Convert flow name to prompt task name that NeMo Guardrails expects
            task_name = flow_name.replace(" ", "_")

            # Check if we have a prompt for this flow (try underscore format first, then spaces)
            if task_name in all_prompts:
                needed_prompts[task_name] = all_prompts[task_name]
                logger.debug("Found prompt for task '%s' using underscore format", task_name)
            elif flow_name in all_prompts:
                needed_prompts[task_name] = all_prompts[flow_name]
                logger.debug("Found prompt for task '%s' using space format", task_name)
            else:
                logger.warning("No prompt found for flow '%s' (task '%s')", flow_name, task_name)

        logger.debug("Needed prompts: %s", list(needed_prompts.keys()))
        logger.debug("Available prompts: %s", list(all_prompts.keys()))

        if needed_prompts:
            yaml_config["prompts"] = []
            for task_name, prompt_content in needed_prompts.items():
                # Ensure the prompt content is properly formatted for YAML literal block scalar
                # Add newlines and proper spacing to match the working format
                formatted_content = prompt_content.strip() + "\n\n"

                prompt_entry = {"task": task_name, "content": formatted_content}
                yaml_config["prompts"].append(prompt_entry)
                logger.debug("Added prompt for task '%s': %s", task_name, prompt_content[:100])
        else:
            logger.warning("No prompts needed or found")

        # Generate Colang content
        colang_content = ""

        # Add bot message definitions
        for message_name, message_content in self.bot_messages.items():
            bot_def_name = message_name.replace("_", " ")
            colang_content += f'\ndefine bot {bot_def_name}\n    "{message_content}"\n'

        # Add all custom Colang flows found in the configuration

        for flow_name in all_flow_names:
            flow_key = flow_name.replace(" ", "_")  # Convert to underscore version
            if flow_key in all_flows:
                logger.debug("Adding custom Colang flow for %s (key: %s)", flow_name, flow_key)
                colang_content += "\n" + all_flows[flow_key]

        logger.debug("colang_content: %s", colang_content)

        return {
            "yaml_content": yaml_config,
            "colang_content": colang_content.strip(),
        }

    def get_available_flows(self) -> dict[str, list[str]]:
        """Get all available default flows organized by type."""
        input_flows = []
        output_flows = []

        for flow_name in self.colang_flows.keys():
            if "input" in flow_name:
                input_flows.append(flow_name)
            elif "output" in flow_name:
                output_flows.append(flow_name)

        return {
            "input": input_flows,
            "output": output_flows,
            "all": list(self.colang_flows.keys()),
        }

    def validate_flow_exists(self, flow_name: str) -> bool:
        """Check if a flow exists in the defaults."""
        return flow_name in self.colang_flows

    def get_flow_dependencies(self, flow_name: str) -> list[str]:
        """Get dependencies for a specific flow (e.g., required bot messages)."""
        dependencies = []

        if flow_name == "self_check_input":
            dependencies.append("refuse_to_respond")
        elif flow_name == "self_check_output":
            dependencies.append("inform_cannot_respond")
        elif flow_name == "self_check_facts":
            dependencies.append("inform_cannot_verify_facts")
        elif flow_name == "self_check_hallucination":
            dependencies.append("inform_potential_hallucination")

        return dependencies
