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
    pass


@register_function(config_type=LocalEventsToolConfig)
async def local_events(tool_config: LocalEventsToolConfig, builder: Builder):

    async def _local_events(city: str) -> LocalEventsResponse:
        events_data = [{
            "event": "Cherry Blossom Tour", "cost": 40
        }, {
            "event": "Modern Art Expo", "cost": 30
        }, {
            "event": "Sushi Making Workshop", "cost": 50
        }, {
            "event": "Vegan Food Festival", "cost": 20
        }, {
            "event": "Vegan Michelin Star Restaurant", "cost": 100
        }]
        events = []
        for event in events_data:
            events.append(LocalEvent(name=event["event"], cost=event["cost"], city=city))
        return LocalEventsResponse(events=events)

    yield FunctionInfo.from_fn(
        _local_events,
        description=("This tool can provide information and cost of local events and activities in a city"))
