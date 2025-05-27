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

import logging

from langchain_core.agents import AgentAction

from aiq.data_models.intermediate_step import IntermediateStep
from aiq.data_models.intermediate_step import IntermediateStepType

logger = logging.getLogger(__name__)


class IntermediateStepAdapter:
    DEFAULT_EVENT_FILTER = [IntermediateStepType.LLM_END, IntermediateStepType.TOOL_END]

    def filter_intermediate_steps(self,
                                  intermediate_steps: list[IntermediateStep],
                                  event_filter: list[IntermediateStepType]) -> list[IntermediateStep]:
        """ Filters intermediate steps"""
        if not event_filter:
            return intermediate_steps
        return [step for step in intermediate_steps if step.event_type in event_filter]

    def validate_intermediate_steps(self, intermediate_steps: list[dict]) -> list[IntermediateStep]:
        validated_steps = []
        for step_data in intermediate_steps:
            try:
                validated_steps.append(IntermediateStep.model_validate(step_data))
            except Exception as e:
                logger.exception("Validation failed for step: %r, Error: %s", step_data, e, exc_info=True)
        return validated_steps

    def serialize_intermediate_steps(self, intermediate_steps: list[IntermediateStep]) -> list[dict]:
        """Converts a list of IntermediateStep objects to a list of dictionaries."""
        return [step.model_dump() for step in intermediate_steps]

    @staticmethod
    def agent_action_to_dict(action) -> dict:
        """Convert AgentAction to a JSON-serializable dictionary."""
        return {
            "tool": action.tool,
            "tool_input": action.tool_input,
            "log": action.log,
            "type": action.type,
        }

    def get_agent_action_single(self, step: IntermediateStep,
                                last_llm_end_step: IntermediateStep | None) -> tuple[AgentAction, str]:
        """Converts a single intermediate step to Tuple[AgentAction, str]."""
        # use the previous llm output as log
        log = getattr(last_llm_end_step.data, "output", "") if last_llm_end_step else ""
        tool_name = step.name or ""
        tool_input = getattr(step.data, "input", "") if step.data else ""
        tool_output = getattr(step.data, "output", "") if step.data else ""

        action = AgentAction(tool=tool_name, tool_input=tool_input, log=log)

        return action, tool_output

    def get_agent_actions(self, intermediate_steps: list[IntermediateStep],
                          event_filter: list[IntermediateStepType]) -> list[tuple[AgentAction, str]]:
        """Converts a list of intermediate steps to a list of (AgentAction, output)."""
        steps = self.filter_intermediate_steps(intermediate_steps, event_filter)
        last_llm_end_step = None
        agent_actions = []
        for step in steps:
            if step.event_type == IntermediateStepType.LLM_END:
                last_llm_end_step = step
                action = self.get_agent_action_single(step, "")
                agent_actions.append(action)
            else:
                action = self.get_agent_action_single(step, last_llm_end_step)
                agent_actions.append(action)

        return agent_actions

    def get_context(self, intermediate_steps: list[IntermediateStep],
                    event_filter: list[IntermediateStepType]) -> list[str]:
        """Grab the output of all the tools and return them as retrieved context."""
        count = 0
        agent_actions = []
        for step in intermediate_steps:
            if step.event_type in event_filter and step.data and step.data.output:
                agent_actions.append(f"**Step {count}**\n{str(step.data.output)}")
                count += 1
        return agent_actions
