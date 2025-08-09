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

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.context import Context
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.interactive import HumanPromptText
from aiq.data_models.interactive import InteractionResponse

logger = logging.getLogger(__name__)


class HITLApprovalFnConfig(FunctionBaseConfig, name="hitl_approval_tool"):
    """
    This function is used to get the user's response to the prompt.
    It will return True if the user responds with 'yes', otherwise False.
    """

    prompt: str = Field(..., description="The prompt to use for the HITL function")


@register_function(config_type=HITLApprovalFnConfig)
async def hitl_approval_function(config: HITLApprovalFnConfig, builder: Builder):

    import re

    prompt = f"{config.prompt} Please confirm if you would like to proceed. Respond with 'yes' or 'no'."

    async def _arun(unused: str = "") -> bool:

        aiq_context = Context.get()
        user_input_manager = aiq_context.user_interaction_manager

        human_prompt_text = HumanPromptText(text=prompt, required=True, placeholder="<your response here>")
        response: InteractionResponse = await user_input_manager.prompt_user_input(human_prompt_text)
        response_str = response.content.text.lower()  # type: ignore
        selected_option = re.search(r'\b(yes)\b', response_str)

        if selected_option:
            return True
        return False

    yield FunctionInfo.from_fn(_arun,
                               description=("This function will be used to get the user's response to the prompt"))
