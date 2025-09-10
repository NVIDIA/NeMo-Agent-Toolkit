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

import sys
from collections.abc import Mapping
from enum import Enum
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field

if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

TagMappingT = TypeVar("TagMappingT", bound=Mapping)


class BaseTaggingConfigMixin(BaseModel, Generic[TagMappingT]):
    """Base mixin for tagging spans."""
    tags: TagMappingT | None = Field(default=None, description="Tags to add to the span.")


class PrivacyLevel(str, Enum):
    """Privacy level for the traces."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


PrivacyTagSchema = TypedDict(
    "PrivacyTagSchema",
    {
        "privacy.level": PrivacyLevel,
    },
    total=True,
)


class PrivacyTaggingConfigMixin(BaseTaggingConfigMixin[PrivacyTagSchema]):
    """Mixin for privacy level tagging on spans."""
    pass


class CustomTaggingConfigMixin(BaseTaggingConfigMixin[dict[str, str]]):
    """Mixin for string key-value tagging on spans."""
    pass
