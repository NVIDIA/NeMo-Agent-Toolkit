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

import logging
from enum import Enum

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ActionEnum(str, Enum):
    PUBLISH = "publish"
    PULL = "pull"
    REMOVE = "remove"
    SEARCH = "search"


class StatusEnum(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class StatusMessage(BaseModel):
    """Represents a data model to record success or error when performing registry interactions.

    Args:
        status (StatusEnum): Represents the outcome (success or error) of the registry interaction.
        action: (ActionEnum): Represents the type of registry action that was taken.
        message: (str): Provides a more detailed status message for the registry interaction.
    """

    status: StatusEnum
    action: ActionEnum
    message: str
