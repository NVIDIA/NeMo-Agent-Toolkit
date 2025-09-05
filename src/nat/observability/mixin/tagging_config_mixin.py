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

from enum import Enum
from typing import Generic
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field

TagValueT = TypeVar("TagValueT")


class PrivacyLevel(str, Enum):
    """Privacy level for the traces."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaggingConfigMixin(BaseModel, Generic[TagValueT]):
    """Generic mixin for tagging spans with typed values.

    This mixin provides a flexible tagging system where both the tag key
    and value type can be customized for different use cases.
    """
    tag_key: str | None = Field(default=None, description="Key to use when tagging traces.")
    tag_value: TagValueT | None = Field(default=None, description="Value to tag the traces with.")


class PrivacyTaggingConfigMixin(TaggingConfigMixin[PrivacyLevel]):
    """Mixin for privacy level tagging on spans.

    Specializes TaggingConfigMixin to work with PrivacyLevel enum values,
    providing a typed interface for privacy-related span tagging.
    """
    pass
