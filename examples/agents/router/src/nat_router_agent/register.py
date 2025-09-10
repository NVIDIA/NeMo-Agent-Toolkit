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

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class MockFruitAdvisorFunctionConfig(FunctionBaseConfig, name="mock_fruit_advisor"):
    pass


@register_function(config_type=MockFruitAdvisorFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_fruit_advisor_function(config: MockFruitAdvisorFunctionConfig, builder: Builder):
    """
    Create a mock fruit advisor function that recommends a fruit based on the input

    Parameters
    ----------
    config : MockFruitAdvisorFunctionConfig
        Configuration for the mock fruit advisor function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that can generate mock fruit advisor based on the input
    """

    async def fruit_advisor(input: str) -> str:
        if "yellow" in input.lower():
            return "banana"
        elif "red" in input.lower():
            return "apple"
        elif "green" in input.lower():
            return "pear"
        else:
            return "I don't know what fruit you are talking about"

    yield FunctionInfo.from_fn(fruit_advisor, description="recommend a fruit based on the input")


class MockCityAdvisorFunctionConfig(FunctionBaseConfig, name="mock_city_advisor"):
    pass


@register_function(config_type=MockCityAdvisorFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_city_advisor_function(config: MockCityAdvisorFunctionConfig, builder: Builder):
    """
    Create a mock city advisor function that recommends a city based on the input

    Parameters
    ----------
    config : MockCityAdvisorFunctionConfig
        Configuration for the mock city advisor function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that can generate mock city advisor based on the input
    """

    async def city_advisor(input: str) -> str:
        if "united states" in input.lower():
            return "New York"
        elif "united kingdom" in input.lower():
            return "London"
        elif "canada" in input.lower():
            return "Toronto"
        elif "australia" in input.lower():
            return "Sydney"
        elif "india" in input.lower():
            return "Mumbai"
        else:
            return "I don't know what city you are talking about"

    yield FunctionInfo.from_fn(city_advisor, description="recommend a city based on the input")


class MockLiteratureAdvisorFunctionConfig(FunctionBaseConfig, name="mock_literature_advisor"):
    pass


@register_function(config_type=MockLiteratureAdvisorFunctionConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def mock_literature_advisor_function(config: MockLiteratureAdvisorFunctionConfig, builder: Builder):
    """
    Create a mock literature advisor function that recommends a literature based on the input

    Parameters
    ----------
    config : MockLiteratureAdvisorFunctionConfig
        Configuration for the mock literature advisor function
    builder : Builder
        The NAT builder instance

    Returns
    -------
    A FunctionInfo object that can generate mock literature advisor based on the input
    """

    async def literature_advisor(input: str) -> str:
        if "shakespeare" in input.lower():
            return "Hamlet"
        elif "dante" in input.lower():
            return "The Divine Comedy"
        elif "milton" in input.lower():
            return "Paradise Lost"
        else:
            return "I don't know what literature you are talking about"

    yield FunctionInfo.from_fn(literature_advisor, description="recommend a literature based on the input")
