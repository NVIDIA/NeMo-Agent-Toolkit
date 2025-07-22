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

from pydantic import BaseModel

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig


class LocalEvent(BaseModel):
    name: str
    cost: float
    city: str


class LocalEventsResponse(BaseModel):
    events: list[LocalEvent]


class LocalEventsToolConfig(FunctionBaseConfig, name="local_events"):
    data_path: str = "examples/frameworks/semantic_kernel_demo/data/local_events.json"


@register_function(config_type=LocalEventsToolConfig)
async def local_events(tool_config: LocalEventsToolConfig, builder: Builder):

    import json

    with open(tool_config.data_path, "r") as f:
        events = LocalEventsResponse.model_validate({"events": json.load(f)}).events

    async def _local_events(city: str) -> LocalEventsResponse:
        return LocalEventsResponse(events=[e for e in events if e.city == city])

    yield FunctionInfo.from_fn(
        _local_events,
        description=("This tool can provide information and cost of local events and activities in a city"))
