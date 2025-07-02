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

import typing

from pydantic import BaseModel


class UsageStatsLLM(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class UsageStatsItem(BaseModel):
    usage_stats_per_llm: dict[str, UsageStatsLLM]
    total_tokens: int | None = None
    runtime: float = 0.0
    min_timestamp: float = 0.0
    max_timestamp: float = 0.0
    llm_latency: float = 0.0


class UsageStats(BaseModel):
    # key is the id or input_obj from EvalInputItem
    min_timestamp: float = 0.0
    max_timestamp: float = 0.0
    total_runtime: float = 0.0
    usage_stats_items: dict[typing.Any, UsageStatsItem] = {}
