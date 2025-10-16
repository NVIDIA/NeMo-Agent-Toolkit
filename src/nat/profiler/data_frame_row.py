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

from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator


class DataFrameRow(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    event_type: Any
    event_timestamp: float | None
    example_number: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    llm_text_input: str | None
    llm_text_output: str | None
    llm_new_token: str | None
    llm_name: str | None
    tool_name: str | None
    function_name: str | None
    function_id: str | None
    parent_function_name: str | None
    parent_function_id: str | None
    UUID: str | None
    framework: str | None

    @field_validator('llm_text_input', 'llm_text_output', 'llm_new_token', mode='before')
    def cast_to_str(cls, v):
        if v is None:
            return v
        try:
            return str(v)
        except Exception as e:
            raise ValueError(f"Value {v} cannot be cast to str: {e}") from e
