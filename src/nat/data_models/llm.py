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

import typing
from enum import Enum

from pydantic import Field

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class APITypeEnum(str, Enum):
    CHAT_COMPLETION = "chat_completion"
    RESPONSES = "responses"


class LLMBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration for LLM providers."""

    api_type: APITypeEnum = Field(default=APITypeEnum.CHAT_COMPLETION,
                                  description="The type of API to use for the LLM provider.",
                                  json_schema_extra={
                                      "enum": [e.value for e in APITypeEnum],
                                      "examples": [e.value for e in APITypeEnum],
                                  })


LLMBaseConfigT = typing.TypeVar("LLMBaseConfigT", bound=LLMBaseConfig)
