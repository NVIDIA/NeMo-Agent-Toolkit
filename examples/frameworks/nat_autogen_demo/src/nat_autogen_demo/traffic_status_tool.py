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
"""Los Angeles traffic status tool file with time-of-day awareness."""

from collections.abc import AsyncIterator

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class TrafficStatusToolConfig(FunctionBaseConfig, name="traffic_status_autogen"):
    """Configuration for the traffic status tool."""


def _get_time_period(hour: int) -> str:
    """Categorize hour into traffic period.

    Args:
        hour: Hour of day (0-23).

    Returns:
        Traffic period: 'morning_rush', 'evening_rush', or 'off_peak'.
    """
    if 7 <= hour <= 9:
        return "morning_rush"
    if 16 <= hour <= 19:
        return "evening_rush"
    return "off_peak"


# Traffic data organized by highway, direction, and time period
TRAFFIC_DATA = {
    "405-south": {
        "morning_rush": "Traffic on the 405 South is heavy between Mulholland Drive and LAX due to morning commuters.",
        "evening_rush": "Traffic on the 405 South is light between Mulholland Drive and LAX as commuters head north.",
        "off_peak": "Traffic on the 405 South is light between Mulholland Drive and LAX.",
    },
    "405-north": {
        "morning_rush":
            "Traffic on the 405 North is light between Westchester and Culver City.",
        "evening_rush":
            "Traffic on the 405 North is heavy between Westchester and Culver City due to evening commuters.",
        "off_peak":
            "Traffic on the 405 North is light between Westchester and Culver City.",
    },
    "110-south": {
        "morning_rush": "Traffic on the 110 South is heavy between Dodger Stadium and Downtown LA due to morning rush.",
        "evening_rush": "Traffic on the 110 South is light from Downtown LA toward Long Beach.",
        "off_peak": "Traffic on the 110 South is light from Pasadena to Long Beach.",
    },
    "110-north": {
        "morning_rush": "Traffic on the 110 North is light from Long Beach to Pasadena.",
        "evening_rush": "Traffic on the 110 North is heavy from Downtown LA toward Pasadena due to evening commuters.",
        "off_peak": "Traffic on the 110 North is light from Long Beach to Pasadena.",
    },
    "10-east": {
        "morning_rush": "Traffic on the 10 East is heavy from Santa Monica to Downtown LA due to morning commuters.",
        "evening_rush": "Traffic on the 10 East is light from Santa Monica to East Los Angeles.",
        "off_peak": "Traffic on the 10 East is light from Santa Monica to East Los Angeles.",
    },
    "10-west": {
        "morning_rush": "Traffic on the 10 West is light from Downtown LA toward Santa Monica.",
        "evening_rush": "Traffic on the 10 West is heavy from Downtown LA to Santa Monica due to evening commuters.",
        "off_peak": "Traffic on the 10 West is light from East LA to Santa Monica.",
    },
    "210-east": {
        "morning_rush": "Traffic on the 210 East is heavy from Pasadena to Azusa due to morning commuters.",
        "evening_rush": "Traffic on the 210 East is light from Pasadena to Azusa.",
        "off_peak": "Traffic on the 210 East is light from Pasadena to Azusa.",
    },
    "210-west": {
        "morning_rush": "Traffic on the 210 West is light from Azusa toward Pasadena.",
        "evening_rush": "Traffic on the 210 West is heavy from Azusa to Pasadena due to evening commuters.",
        "off_peak": "Traffic on the 210 West is light from Azusa to Pasadena.",
    },
}


@register_function(config_type=TrafficStatusToolConfig, framework_wrappers=[LLMFrameworkEnum.AUTOGEN])
async def traffic_status(_config: TrafficStatusToolConfig, _builder: Builder) -> AsyncIterator[FunctionInfo]:
    """NAT function that provides traffic status for Los Angeles based on time of day.

    Args:
        _config (TrafficStatusToolConfig): The configuration for the traffic status tool.
        _builder (Builder): The NAT builder instance.

    Yields:
        AsyncIterator[FunctionInfo]: Yields a FunctionInfo object encapsulating the traffic status tool.
    """

    async def _traffic_status(hwy: str, hour: int) -> str:
        """
        Get the traffic status for a Los Angeles highway at a specific hour.

        Args:
            hwy (str): The highway name and direction. Supported highways:
                '405-south', '405-north', '110-south', '110-north',
                '10-east', '10-west', '210-east', '210-west'.
            hour (int): The hour of day (0-23). Use the current hour from the datetime tool.

        Returns:
            str: The traffic status for the specified highway and hour.
        """
        # Parse the highway
        hwy_lower = hwy.lower().strip()
        hwy_key = None
        for key in TRAFFIC_DATA:
            if key in hwy_lower:
                hwy_key = key
                break

        if hwy_key is None:
            return (f"Traffic information for '{hwy}' is not available. "
                    f"Supported highways: 405-south, 405-north, 110-south, 110-north, "
                    f"10-east, 10-west, 210-east, 210-west.")

        # Validate hour
        if not 0 <= hour <= 23:
            return f"Invalid hour '{hour}'. Please provide an hour between 0 and 23."

        # Get traffic period and return appropriate message
        period = _get_time_period(hour)
        traffic_info = TRAFFIC_DATA[hwy_key][period]

        return traffic_info

    yield FunctionInfo.from_fn(_traffic_status, description=_traffic_status.__doc__)
