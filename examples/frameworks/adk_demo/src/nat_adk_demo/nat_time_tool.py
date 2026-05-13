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

import datetime
import logging
from collections.abc import AsyncIterator
from urllib.error import URLError
from zoneinfo import ZoneInfo
from zoneinfo import ZoneInfoNotFoundError

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .location import DEFAULT_GEOCODING_URL
from .location import DEFAULT_TIMEOUT_SECONDS
from .location import CityLocation
from .location import geocode_city

logger = logging.getLogger(__name__)


class TimeMCPToolConfig(FunctionBaseConfig, name="get_city_time_tool"):
    """Configuration for the get_city_time tool."""

    geocoding_url: str = Field(default=DEFAULT_GEOCODING_URL, description="Open-Meteo geocoding API URL.")
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0, description="HTTP request timeout.")


def _format_city_time(location: CityLocation, now: datetime.datetime) -> str:
    return f"The current time in {location.display_name} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}."


async def get_time_for_city(city: str, config: TimeMCPToolConfig) -> str:
    """Get the local time for a city using geocoding metadata."""

    try:
        location = await geocode_city(city, config.geocoding_url, config.timeout_seconds)
        if location is None:
            return f"Sorry, I don't have timezone information for {city} because the city could not be found."

        now = datetime.datetime.now(ZoneInfo(location.timezone))
    except (TimeoutError, URLError, ValueError, OSError) as ex:
        return f"Sorry, I could not retrieve timezone information for {city}: {ex}"
    except ZoneInfoNotFoundError:
        return f"Sorry, I don't have usable timezone information for {city}."

    return _format_city_time(location, now)


@register_function(config_type=TimeMCPToolConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def get_city_time(config: TimeMCPToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:
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

        return await get_time_for_city(city, config)

    yield FunctionInfo.from_fn(_get_city_time, description=_get_city_time.__doc__)
