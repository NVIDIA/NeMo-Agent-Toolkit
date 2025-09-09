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

from pydantic import BaseModel
from pydantic import Field


class BatchConfigMixin(BaseModel):
    """Mixin for telemetry exporters that require batching."""
    batch_size: int = Field(default=100, description="The batch size for the telemetry exporter.")
    flush_interval: float = Field(default=5.0, description="The flush interval for the telemetry exporter.")
    max_queue_size: int = Field(default=1000, description="The maximum queue size for the telemetry exporter.")
    drop_on_overflow: bool = Field(default=False, description="Whether to drop on overflow for the telemetry exporter.")
    shutdown_timeout: float = Field(default=10.0, description="The shutdown timeout for the telemetry exporter.")
