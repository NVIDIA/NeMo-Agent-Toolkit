# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import zoneinfo

from starlette.datastructures import Headers

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.settings.global_settings import GlobalSettings


class CurrentTimeToolConfig(FunctionBaseConfig, name="current_datetime"):
    """
    Simple tool which returns the current date and time in human readable format with timezone information. By default,
    the timezone is in Etc/UTC. If the user provides a timezone in the header, we will use it. Timezone will be
    provided in IANA zone name format. For example, "America/New_York" or "Etc/UTC".
    """
    pass


def _get_timezone_obj(headers: Headers | None) -> zoneinfo.ZoneInfo | datetime.tzinfo:
    # Default to UTC
    timezone_obj = zoneinfo.ZoneInfo("Etc/UTC")

    if headers:
        # If user has provided a timezone in the header, we will prioritize on using it
        timezone_header = headers.get("x-timezone")
        if timezone_header:
            try:
                timezone_obj = zoneinfo.ZoneInfo(timezone_header)
            except Exception:
                pass
    else:
        # Only if a timezone is not in the header, we will determine default timezone based on global settings
        fallback_tz = GlobalSettings.get().fallback_timezone

        if fallback_tz == "system":
            # Use the system's local timezone. Avoid requiring external deps.
            timezone_obj = datetime.datetime.now().astimezone().tzinfo or zoneinfo.ZoneInfo("Etc/UTC")

    return timezone_obj


@register_function(config_type=CurrentTimeToolConfig)
async def current_datetime(_config: CurrentTimeToolConfig, _builder: Builder):

    async def _get_current_time(unused: str) -> str:

        del unused  # Unused parameter to avoid linting error

        from nat.builder.context import Context
        nat_context = Context.get()

        headers: Headers | None = nat_context.metadata.headers

        timezone_obj = _get_timezone_obj(headers)

        now = datetime.datetime.now(timezone_obj)
        now_machine_readable = now.strftime("%Y-%m-%d %H:%M:%S %z")

        # Returns the current time in machine readable format with timezone offset.
        return f"The current time of day is {now_machine_readable}"

    yield FunctionInfo.from_fn(
        _get_current_time,
        description="Returns the current date and time in human readable format with timezone information.")
