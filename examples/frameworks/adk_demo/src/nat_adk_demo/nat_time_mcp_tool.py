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
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class TimeMCPToolConfig(FunctionBaseConfig, name="get_city_time_tool"):
    """Configuration for the get_city_time tool."""


@register_function(config_type=TimeMCPToolConfig)
async def get_city_time(_config: TimeMCPToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:
    """Register the get_city_time(city: str) -> str MCP tool.

    Yields:
        FunctionInfo: Descriptor for an async function `_get_city_time(city: str) -> str`.
    """

    async def _get_city_time(city: str) -> str:
        """Get the time in a specified city.
        Currently supports New York.
        Args:
            city (str): The name of the city.

        Returns:
            str: The current time in the specified city or an error message if the city is not recognized.
        """

        if city.strip().casefold() in {"new york", "new york city", "nyc"}:
            tz_identifier = "America/New_York"
        else:
            return f"Sorry, I don't have timezone information for {city}."

        tz = ZoneInfo(tz_identifier)
        now = datetime.datetime.now(tz)
        report = f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
        return report

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(_get_city_time,
                               description=("This tool provides the current time in a specified city. "
                                            "It takes a city name as input and returns the current time in that city."))
