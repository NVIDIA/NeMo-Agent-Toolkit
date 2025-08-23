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

import json

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator


class SWEBenchInput(BaseModel):
    # Allow extra fields in the model_config to support derived models
    model_config = ConfigDict(extra="allow")

    repo: str
    instance_id: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str | int
    version: float
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    environment_setup_commit: str

    # Handle improperly formatted JSON strings for list fields
    @field_validator("FAIL_TO_PASS", "PASS_TO_PASS", mode="before")
    def parse_list_fields(cls, value):
        if isinstance(value, str):
            # Attempt to parse the string as a list
            return json.loads(value)
        return value


class SWEBenchOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), populate_by_name=True)

    instance_id: str
    model_name_or_path: str
    model_patch: str
