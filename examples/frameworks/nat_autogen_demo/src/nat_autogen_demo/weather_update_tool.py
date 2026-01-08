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
"""Weather update tool file."""

from collections.abc import AsyncIterator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class WeatherToolConfig(FunctionBaseConfig, name="weather_update_autogen"):
    """Configuration for the weather update tool."""


@register_function(config_type=WeatherToolConfig, framework_wrappers=[LLMFrameworkEnum.AUTOGEN])
async def weather_update(_config: WeatherToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:
    """NAT function that provides weather updates for a specified city.

    Args:
        _config (WeatherToolConfig): The configuration for the weather update tool.
        _builder (Builder): The NAT builder instance.

    Yields:
        AsyncIterator[FunctionInfo]: Yields a FunctionInfo object encapsulating the weather update tool
    """

    async def _weather_update(city: str) -> str:
        """
        Get the current weather for a specified city.

        Args:
            city (str): The name of the city.

        Returns:
            str: The current weather for the specified city.
        """
        city_lower = city.lower()
        if "new york" in city_lower:
            return ("The weather in New York is sunny with a temperature "
                    "of 25 degrees Celsius (77 degrees Fahrenheit).")
        if "london" in city_lower:
            return ("The weather in London is cloudy with a temperature "
                    "of 15 degrees Celsius (59 degrees Fahrenheit).")
        if "tokyo" in city_lower:
            return ("The weather in Tokyo is partly cloudy with a temperature "
                    "of 22 degrees Celsius (72 degrees Fahrenheit).")
        if "paris" in city_lower:
            return ("The weather in Paris is rainy with a temperature "
                    "of 18 degrees Celsius (64 degrees Fahrenheit).")
        if "san francisco" in city_lower or "sf" in city_lower:
            return ("The weather in San Francisco is foggy with a temperature "
                    "of 16 degrees Celsius (61 degrees Fahrenheit).")
        return f"Weather information for '{city}' is not available."

    yield FunctionInfo.from_fn(_weather_update, description=_weather_update.__doc__)
