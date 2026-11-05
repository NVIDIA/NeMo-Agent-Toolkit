# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class TokenUsageBaseModel(BaseModel):
    """Shared token usage counters emitted by LLM/tool events."""

    prompt_tokens: int = Field(default=0, description="Number of tokens in the prompt.")
    completion_tokens: int = Field(default=0, description="Number of tokens in the completion.")
    cached_tokens: int = Field(default=0, description="Number of tokens read from cache.")
    reasoning_tokens: int = Field(default=0, description="Number of tokens used for reasoning.")
    total_tokens: int = Field(default=0, description="Number of tokens total.")

    @model_validator(mode="before")
    @classmethod
    def _flatten_nested_details(cls, data: Any) -> Any:
        """Read cached/reasoning counts from OpenAI-style nested usage details when the flat fields are absent."""
        if not isinstance(data, dict):
            return data

        for flat, details_key, nested in (("cached_tokens", "prompt_tokens_details", "cached_tokens"),
                                          ("reasoning_tokens", "completion_tokens_details", "reasoning_tokens")):
            details = data.get(details_key)
            if data.get(flat) is None and isinstance(details, dict) and details.get(nested) is not None:
                data = {**data, flat: details[nested]}
        return data
