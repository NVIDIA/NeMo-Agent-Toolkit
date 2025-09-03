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

from pydantic import BaseModel
from pydantic import Field


class PrivacyLevel(Enum):
    """Privacy level for the traces."""
    NONE = "none"
    BASIC = "basic"
    MEDIUM = "medium"
    HIGH = "high"


class RedactionConfigMixin(BaseModel):
    """Mixin for telemetry exporters that require redaction configuration."""
    redaction_enabled: bool = Field(default=False, description="Whether to redact PII from the traces.")
    redaction_attributes: list[str] = Field(default_factory=lambda: ["input.value", "output.value", "metadata"],
                                            description="Attributes to redact from the traces.")
    redaction_header: str = Field(default="x-redaction-key", description="Header to check for redaction.")
    tag_key: str = Field(default="privacy_level", description="Key to tag the traces.")
    tag_value: PrivacyLevel = Field(default=PrivacyLevel.NONE, description="Value to tag the traces.")
