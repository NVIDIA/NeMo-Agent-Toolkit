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

import asyncio
import logging
from datetime import datetime

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.workflow import Workflow

logger = logging.getLogger(__name__)


class UserWorkflowData(BaseModel):
    """Data model for per-user workflow data.

    This model is used to store data for a user's workflow.
    """
    user_id: str = Field(description="The ID of the user.")
    workflow: Workflow = Field(description="The workflow for the user.")
    builder: Builder = Field(description="The workflow builder for the user.")
    last_activity: datetime = Field(description="The last activity time for the user's workflow.")
    ref_count: int = Field(description="The reference count for the user's workflow.")
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, description="The lock for the user's workflow.")
    stop_event: asyncio.Event = Field(default_factory=asyncio.Event,
                                      description="The stop event for the user's workflow.")
    lifetime_task: asyncio.Task | None = Field(default=None, description="The lifetime task for the user's workflow.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
