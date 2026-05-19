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

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from urllib.error import URLError

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from .location import DEFAULT_GEOCODING_URL
from .location import DEFAULT_TIMEOUT_SECONDS
from .location import CityLocation
from .location import fetch_json
from .location import geocode_city

DEFAULT_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_CODE_DESCRIPTIONS = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherToolConfig(FunctionBaseConfig, name="weather_update"):
    """Configuration for the weather update tool."""

    geocoding_url: str = Field(default=DEFAULT_GEOCODING_URL, description="Open-Meteo geocoding API URL.")
    forecast_url: str = Field(default=DEFAULT_FORECAST_URL, description="Open-Meteo forecast API URL.")
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, gt=0, description="HTTP request timeout.")


async def _fetch_current_weather(location: CityLocation, config: WeatherToolConfig) -> dict[str, Any]:
    params = {
        "latitude":
            location.latitude,
        "longitude":
            location.longitude,
        "current":
            ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "precipitation",
                "weather_code",
                "wind_speed_10m",
            ]),
        "temperature_unit":
            "celsius",
        "wind_speed_unit":
            "kmh",
        "timezone":
            "auto",
    }
    return await asyncio.to_thread(fetch_json, config.forecast_url, params, config.timeout_seconds)


def _format_number(value: Any) -> str | None:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.1f}".rstrip("0").rstrip(".")
    return None


def _format_weather(location: CityLocation, payload: dict[str, Any]) -> str:
    current = payload.get("current")
    if not isinstance(current, dict):
        return f"Weather information for {location.display_name} is not available right now."

    temperature = _format_number(current.get("temperature_2m"))
    apparent_temperature = _format_number(current.get("apparent_temperature"))
    humidity = _format_number(current.get("relative_humidity_2m"))
    precipitation = _format_number(current.get("precipitation"))
    wind_speed = _format_number(current.get("wind_speed_10m"))
    try:
        weather_code = int(current["weather_code"])
    except (KeyError, TypeError, ValueError):
        weather_code = None
    description = WEATHER_CODE_DESCRIPTIONS.get(weather_code, "current conditions")

    details = []
    if temperature is not None:
        details.append(f"{temperature} degrees Celsius")
    if apparent_temperature is not None:
        details.append(f"feels like {apparent_temperature} degrees Celsius")
    if humidity is not None:
        details.append(f"{humidity}% humidity")
    if precipitation is not None:
        details.append(f"{precipitation} mm precipitation")
    if wind_speed is not None:
        details.append(f"{wind_speed} km/h wind")

    if details:
        return f"The current weather in {location.display_name} is {description}, with {', '.join(details)}."

    return f"The current weather in {location.display_name} is {description}."


async def get_current_weather(city: str, config: WeatherToolConfig) -> str:
    """Get current weather for a city using geocoding and forecast APIs."""

    try:
        location = await geocode_city(city, config.geocoding_url, config.timeout_seconds)
        if location is None:
            return f"Weather information for '{city}' is not available because the city could not be found."

        payload = await _fetch_current_weather(location, config)
    except (TimeoutError, URLError, ValueError, OSError) as ex:
        return f"Weather information for '{city}' is not available right now: {ex}"

    return _format_weather(location, payload)


@register_function(config_type=WeatherToolConfig, framework_wrappers=[LLMFrameworkEnum.ADK])
async def weather_update(config: WeatherToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:

    async def _weather_update(city: str) -> str:
        """
        Get the current weather for a specified city.

        Args:
            city (str): The name of the city.

        Returns:
            str: The current weather for the specified city.
        """
        return await get_current_weather(city, config)

    yield FunctionInfo.from_fn(_weather_update, description=_weather_update.__doc__)
