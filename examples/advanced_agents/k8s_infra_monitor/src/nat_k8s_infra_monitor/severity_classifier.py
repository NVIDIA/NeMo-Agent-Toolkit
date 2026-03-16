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

"""Tool for classifying the severity of Kubernetes infrastructure incidents."""

import typing

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from nat.plugins.profiler.decorators.function_tracking import track_function

from .prompts import SEVERITY_CLASSIFIER_PROMPT

if typing.TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


class SeverityClassifierConfig(FunctionBaseConfig, name="severity_classifier"):
    """Configuration for the incident severity classifier."""

    llm_name: LLMRef


@register_function(config_type=SeverityClassifierConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def severity_classifier(config: SeverityClassifierConfig, builder: Builder):
    """Classify the severity of a Kubernetes incident report."""

    llm: "BaseChatModel" = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    @track_function()
    async def _run(report: str) -> str:
        """Classify the severity of a diagnostic report as critical, warning, or informational.

        Args:
            report: The full diagnostic report text to classify.

        Returns:
            A severity classification with explanation.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SEVERITY_CLASSIFIER_PROMPT),
            HumanMessage(content=f"Classify the severity of this incident report:\n\n{report}"),
        ]
        response = await llm.ainvoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)
        return f"\n\n## Severity Classification\n{content.strip()}"

    yield _run
