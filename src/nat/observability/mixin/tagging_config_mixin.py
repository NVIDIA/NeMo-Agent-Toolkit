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

import sys
from collections.abc import Mapping
from enum import Enum
from typing import Generic
from typing import TypeVar

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from pydantic import BaseModel
from pydantic import Field

TagDictT = TypeVar("TagDictT", bound=Mapping)


class BaseTaggingConfigMixin(BaseModel, Generic[TagDictT]):
    """Base mixin for tagging spans."""
    tags: TagDictT | None = Field(default=None)


class PrivacyLevel(str, Enum):
    """Privacy level for the traces."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PrivacyTagSchema(TypedDict, total=False):
    """Schema for the tags."""
    privacy_level: PrivacyLevel


class PrivacyTaggingConfigMixin(BaseTaggingConfigMixin[PrivacyTagSchema]):
    """Mixin for privacy level tagging on spans."""


class CustomTaggingConfigMixin(BaseTaggingConfigMixin[dict[str, str]]):
    """Mixin for string key-value tagging on spans."""
