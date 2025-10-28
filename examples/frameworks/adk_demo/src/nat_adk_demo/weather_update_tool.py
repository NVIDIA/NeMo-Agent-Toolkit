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
"""Weather update tool file."""

from collections.abc import AsyncIterator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class WeatherToolConfig(FunctionBaseConfig, name="weather_update"):
    pass


@register_function(config_type=WeatherToolConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def weather_update(_config: WeatherToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:

    async def _weather_update(city: str) -> str:
        """
        Get the current weather for a specified city.

        Args:
            city (str): The name of the city.

        Returns:
            str: The current weather for the specified city.
        """
        if city.lower() == "new york":
            return "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."
        return f"Weather information for '{city}' is not available."

    yield FunctionInfo.from_fn(_weather_update, description=_weather_update.__doc__)
