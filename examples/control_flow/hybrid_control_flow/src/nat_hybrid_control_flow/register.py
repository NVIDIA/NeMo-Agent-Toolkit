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

import logging

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MockInputValidatorFunctionConfig(FunctionBaseConfig, name="mock_input_validator"):
    pass


@register_function(config_type=MockInputValidatorFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_input_validator_function(config: MockInputValidatorFunctionConfig, builder: Builder):
    """
    Create a mock input validator function that validates input

    Parameters
    ----------
    config : MockInputValidatorFunctionConfig
        Configuration for the input validator function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that performs simple input validation
    """

    async def validate(text: str) -> str:
        """Validate input text and add metadata for routing."""
        if not text or len(text.strip()) == 0:
            return "ERROR: Empty input"

        # Add validation metadata
        validated = f"[VALIDATED] {text.strip()}"
        return validated

    yield FunctionInfo.from_fn(validate, description="Validate and prepare input text for processing")


class MockUppercaseConverterFunctionConfig(FunctionBaseConfig, name="mock_uppercase_converter"):
    pass


@register_function(config_type=MockUppercaseConverterFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_uppercase_converter_function(config: MockUppercaseConverterFunctionConfig, builder: Builder):
    """
    Create function that converts text to uppercase

    Parameters
    ----------
    config : MockUppercaseConverterFunctionConfig
        Configuration for the uppercase converter function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that converts text to uppercase
    """

    async def convert_uppercase(text: str) -> str:
        return text.upper()

    yield FunctionInfo.from_fn(convert_uppercase, description="Convert text to uppercase")


class MockLowercaseConverterFunctionConfig(FunctionBaseConfig, name="mock_lowercase_converter"):
    pass


@register_function(config_type=MockLowercaseConverterFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_lowercase_converter_function(config: MockLowercaseConverterFunctionConfig, builder: Builder):
    """
    Create function that converts text to lowercase

    Parameters
    ----------
    config : MockLowercaseConverterFunctionConfig
        Configuration for the lowercase converter function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that converts text to lowercase
    """

    async def convert_lowercase(text: str) -> str:
        return text.lower()

    yield FunctionInfo.from_fn(convert_lowercase, description="Convert text to lowercase")


class MockResultFormatterFunctionConfig(FunctionBaseConfig, name="mock_result_formatter"):
    pass


@register_function(config_type=MockResultFormatterFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_result_formatter_function(config: MockResultFormatterFunctionConfig, builder: Builder):
    """
    Create a mock result formatter function that formats the final output

    Parameters
    ----------
    config : MockResultFormatterFunctionConfig
        Configuration for the result formatter function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that formats the final result
    """

    async def format_result(text: str) -> str:
        """Format the processed result with a wrapper."""
        return f"=== PROCESSED RESULT ===\n{text}\n========================"

    yield FunctionInfo.from_fn(format_result, description="Format the final processing result")
