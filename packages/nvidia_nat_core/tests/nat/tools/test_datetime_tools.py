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

import re
import zoneinfo
from datetime import datetime

import pytest

from nat.test import ToolTestRunner


async def test_current_datetime_tool():
    from nat.tool.datetime_tools import CurrentTimeToolConfig

    expected_result_pattern = r"^The current time of day is (.+)$"

    runner = ToolTestRunner()
    result = await runner.test_tool(config_type=CurrentTimeToolConfig, input_data="unused input")
    assert result is not None

    result_match = re.match(expected_result_pattern, result)
    assert result_match is not None, f"Result '{result}' does not match expected pattern: {expected_result_pattern}."

    datetime_str = result_match.group(1)

    # Validate that the result is a valid datetime string
    try:
        datetime.fromisoformat(datetime_str)
    except ValueError:
        pytest.fail(f"Result '{datetime_str}' is not a datetime string in the expected format.")


async def test_current_timezone_tool():
    from nat.tool.datetime_tools import CurrentTimeZoneToolConfig

    expected_result_pattern = r"^The time zone is (.+)$"

    runner = ToolTestRunner()
    result = await runner.test_tool(config_type=CurrentTimeZoneToolConfig, input_data="unused input")
    assert result is not None

    result_match = re.match(expected_result_pattern, result)
    assert result_match is not None, f"Result '{result}' does not match expected pattern: {expected_result_pattern}."

    timezone_str = result_match.group(1)

    # Validate that the result is a valid timezone string
    try:
        zoneinfo.ZoneInfo(timezone_str)
    except Exception:
        pytest.fail(f"Result '{timezone_str}' is not a valid timezone string.")
