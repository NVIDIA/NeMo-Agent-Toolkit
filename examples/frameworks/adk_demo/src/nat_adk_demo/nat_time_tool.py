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

import datetime
import logging
from collections.abc import AsyncIterator
from zoneinfo import ZoneInfo

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class TimeMCPToolConfig(FunctionBaseConfig, name="get_city_time_tool"):
    """Configuration for the get_city_time tool."""


@register_function(config_type=TimeMCPToolConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def get_city_time(_config: TimeMCPToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:
    """
    Register a get_city_time(city: str) -> str tool for ADK.

    Args:
        _config (TimeMCPToolConfig): The configuration for the get_city_time tool.
        _builder (Builder): The NAT builder instance.
    """

    async def _get_city_time(city: str) -> str:
        """
        Get the time in a specified city.

        Args:
            city (str): The name of the city.

        Returns:
            str: The current time in the specified city or an error message if the city is not recognized.
        """

        if city.strip().casefold() not in {"new york", "new york city", "nyc"}:
            return f"Sorry, I don't have timezone information for {city}."

        now = datetime.datetime.now(ZoneInfo("America/New_York"))
        return f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"

    yield FunctionInfo.from_fn(_get_city_time, description=_get_city_time.__doc__)
