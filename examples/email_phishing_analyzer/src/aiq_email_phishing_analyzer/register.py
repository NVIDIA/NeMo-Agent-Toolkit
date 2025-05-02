# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import logging
from typing import Any

from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from .utils import smart_parse

logger = logging.getLogger(__name__)


class EmailPhishingAnalyzerConfig(FunctionBaseConfig, name="email_phishing_analyzer"):
    _type: str = "email_phishing_analyzer"
    llm: LLMRef  # Name of the LLM to use
    prompt: str


@register_function(config_type=EmailPhishingAnalyzerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def email_phishing_analyzer(config: EmailPhishingAnalyzerConfig, builder: Builder) -> Any:
    """Register the email phishing analysis tool."""

    async def _analyze_email_phishing(text: str) -> str:
        """
        Analyze an email body for signs of phishing using an LLM.

        Args:
            text: The email body text to analyze

        Returns:
            String containing analysis results in a human-readable format
        """
        # Get LLM from builder
        llm = await builder.get_llm(llm_name=config.llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        # Get response from LLM
        response = await llm.apredict(config.prompt.format(body=text))

        try:
            # Parse response using smart_parse
            analysis = smart_parse(response)

            # Handle missing or malformed fields with defaults
            result = {
                "is_likely_phishing": analysis.get('is_likely_phishing', False),
                "explanation": analysis.get('explanation', 'No detailed explanation provided')
            }

            # Return as JSON string
            return json.dumps(result)
        except json.JSONDecodeError:
            return "Error: Could not parse LLM response as JSON"

    # Create a Generic AIQ Toolkit tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(_analyze_email_phishing,
                               description=("This tool analyzes email content to detect signs of phishing "
                                            "attempts. It evaluates factors like urgency, generic greetings, "
                                            "grammar mistakes, unusual requests, and emotional manipulation."))
