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


class RedactionConfigMixin(BaseModel):
    """Mixin for basic redaction configuration.

    Provides core redaction functionality that can be used standalone
    or inherited by specialized redaction mixins.
    """
    redaction_enabled: bool = Field(default=False, description="Whether to enable redaction processing.")
    redaction_value: str = Field(default="[REDACTED]", description="Value to replace redacted attributes with.")
    redaction_attributes: list[str] = Field(default_factory=lambda: ["input.value", "output.value", "nat.metadata"],
                                            description="Attributes to redact when redaction is triggered.")
    force_redaction: bool = Field(default=False, description="Always redact regardless of other conditions.")
    redaction_tag: str | None = Field(default=None, description="Tag to add to spans when redaction is triggered.")


class HeaderRedactionConfigMixin(RedactionConfigMixin):
    """Mixin for header-based redaction configuration.

    Inherits core redaction fields (redaction_enabled, redaction_attributes, force_redaction)
    and adds header-specific configuration for authentication-based redaction decisions.

    Note: The callback function must be provided directly to the processor at runtime.
    """
    redaction_headers: list[str] = Field(default_factory=list, description="Headers to check for redaction decisions.")
