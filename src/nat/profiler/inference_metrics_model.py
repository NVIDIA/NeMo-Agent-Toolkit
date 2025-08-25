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
from pydantic import Field


class InferenceMetricsModel(BaseModel):
    n: int = Field(default=0, description="Number of samples")
    mean: float = Field(default=0, description="Mean of the samples")
    ninetieth_interval: tuple[float, float] = Field(default=(0, 0), description="90% confidence interval")
    ninety_fifth_interval: tuple[float, float] = Field(default=(0, 0), description="95% confidence interval")
    ninety_ninth_interval: tuple[float, float] = Field(default=(0, 0), description="99% confidence interval")
    p90: float = Field(default=0, description="90th percentile of the samples")
    p95: float = Field(default=0, description="95th percentile of the samples")
    p99: float = Field(default=0, description="99th percentile of the samples")
