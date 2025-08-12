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

import asyncio
import logging
import re

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_ttc_strategy
from nat.data_models.ttc_strategy import TTCStrategyBaseConfig
from nat.experimental.test_time_compute.models.editor_config import LLMAsAJudgeEditorConfig
from nat.experimental.test_time_compute.models.stage_enums import PipelineTypeEnum
from nat.experimental.test_time_compute.models.stage_enums import StageTypeEnum
from nat.experimental.test_time_compute.models.strategy_base import StrategyBase
from nat.experimental.test_time_compute.models.ttc_item import TTCItem
from nat.utils.io.model_processing import remove_r1_think_tags

logger = logging.getLogger(__name__)


class LLMAsAJudgeEditor(StrategyBase):
    """
    Given a list of PlanningItems, uses a feedback LLM to generate feedback on each plan
    Then edits the plan based on feedback.
    """

    def __init__(self, config: TTCStrategyBaseConfig) -> None:
        super().__init__(config)
        self.feedback_llm = None
        self.editing_llm = None

    async def build_components(self, builder: Builder) -> None:
        """
        Build the components required for the editor.
        """
        # Get the feedback LLM
        self.feedback_llm = await builder.get_llm(self.config.feedback_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        self.editing_llm = await builder.get_llm(self.config.editing_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    def supported_pipeline_types(self) -> [PipelineTypeEnum]:
        return [PipelineTypeEnum.PLANNING]

    def stage_type(self) -> StageTypeEnum:
        return StageTypeEnum.EDITING

    async def generate_feedback(self, llm, template, context: str, prompt: str, item: TTCItem) -> TTCItem:
        """
        Helper function to generate feedback for a given planning item using the provided prompt.
        """

        prompt = await template.ainvoke(
            input={
                "context": context,
                "original_prompt": prompt,  # Original prompt used to generate the plans
                "plan": item.plan,
                "num_feedback": self.config.num_feedback
            })

        feedback_result = await llm.ainvoke(prompt.to_string())
        if not feedback_result:
            logger.warning(f"No feedback generated for plan: {item.plan}.")
            return item

        # Update the planning item with the generated feedback
        cleaned = remove_r1_think_tags(
            feedback_result.content if hasattr(feedback_result, 'content') else str(feedback_result))

        # Feedback is the string following 'FEEDBACK:'. Use Regex to extract
        cleaned = re.sub(r'(?i)^\s*FEEDBACK:\s*', '', cleaned).strip()
        if not cleaned:
            logger.warning(f"Feedback was empty for plan: {item.plan}.")
            return item

        item.feedback = cleaned  # Set the feedback in the TTCItem

        return item

    async def edit_plan(self, llm, template, context: str, prompt: str, item: TTCItem) -> TTCItem:
        """
        Helper function to edit a plan based on feedback using the provided prompt.
        """

        if not item.feedback:
            logger.warning(f"No feedback available for plan: {item.plan}. Cannot edit.")
            return item

        prompt = await template.ainvoke(
            input={
                "context": context,
                "original_prompt": prompt,  # Original prompt used to generate the plans
                "plan": item.plan,
                "feedback": item.feedback
            })

        editing_result = await llm.ainvoke(prompt.to_string())
        if not editing_result:
            logger.warning(f"No editing result generated for plan: {item.plan}.")
            return item

        # Update the planning item with the edited plan
        cleaned = remove_r1_think_tags(
            editing_result.content if hasattr(editing_result, 'content') else str(editing_result))

        # Plan is the string following 'EDITED PLAN:'. Use Regex to extract
        cleaned = re.sub(r'(?i)^\s*EDITED PLAN:\s*', '', cleaned).strip()
        if not cleaned:
            logger.warning(f"Edited plan was empty for plan: {item.plan}. Returning original.")
            return item

        # Update the plan in the PlanningItem
        item.plan = cleaned

        return item

    async def ainvoke(self,
                      items: list[TTCItem],
                      original_prompt: str | None = None,
                      agent_context: str | None = None,
                      **kwargs) -> list[TTCItem]:
        """
        Edit the provided planning items using a feedback LLM.
        """
        from langchain_core.language_models import BaseChatModel
        from langchain_core.prompts import PromptTemplate

        # assert self.config.feedback_llm is a BaseChatModel
        if not isinstance(self.feedback_llm, BaseChatModel):
            raise ValueError("The `feedback_llm` must be an instance of `BaseChatModel`.")

        # assert self.config.editing_llm is a BaseChatModel
        if not isinstance(self.editing_llm, BaseChatModel):
            raise ValueError("The `editing_llm` must be an instance of `BaseChatModel`.")

        feedback_model: BaseChatModel = self.feedback_llm
        editing_model: BaseChatModel = self.editing_llm

        feedback_template = PromptTemplate(template=self.config.feedback_template,
                                           input_variables=["context", "original_prompt", "plan", "num_feedback"],
                                           validate_template=True)

        editing_template = PromptTemplate(template=self.config.editor_template,
                                          input_variables=["context", "original_prompt", "plan", "feedback"],
                                          validate_template=True)

        # Generate feedback for each planning item concurrently
        feedback_tasks = [
            self.generate_feedback(
                llm=feedback_model,
                template=feedback_template,
                context=agent_context,
                prompt=original_prompt,  # Original prompt used to generate the plans
                item=item) for item in items
        ]
        # Run the feedback tasks concurrently and gather results
        planning_items_with_feedback = await asyncio.gather(*feedback_tasks)

        if not planning_items_with_feedback:
            raise ValueError("No feedback was generated for the planning items. Please check the LLM response.")

        logger.info("Generated feedback for %d plans.", len(planning_items_with_feedback))

        # Now edit each planning item based on the feedback concurrently
        editing_tasks = [
            self.edit_plan(
                llm=editing_model,
                template=editing_template,
                context=agent_context,
                prompt=original_prompt,  # Original prompt used to generate the plans
                item=item) for item in planning_items_with_feedback
        ]
        # Run the editing tasks concurrently and gather results
        edited_planning_items = await asyncio.gather(*editing_tasks)

        if not edited_planning_items:
            raise ValueError("No plans were edited. Please check the LLM response.")

        logger.info("Edited %d plans based on feedback.", len(edited_planning_items))
        return edited_planning_items


@register_ttc_strategy(config_type=LLMAsAJudgeEditorConfig)
async def register_llm_as_a_judge_editor(config: TTCStrategyBaseConfig, builder: Builder):
    """
    Register the LLMAsAJudgeEditor strategy with the provided configuration and builder.
    """

    editor = LLMAsAJudgeEditor(config)
    await editor.build_components(builder)

    yield editor
