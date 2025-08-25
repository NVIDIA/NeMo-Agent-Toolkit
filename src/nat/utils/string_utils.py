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


def convert_to_str(value: Any) -> str:
    """
    Convert a value to a string representation.
    Handles various types including lists, dictionaries, and other objects.
    """
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return ", ".join(map(str, value))
    elif isinstance(value, BaseModel):
        return value.model_dump_json(exclude_none=True, exclude_unset=True)
    elif isinstance(value, dict):
        return ", ".join(f"{k}: {v}" for k, v in value.items())
    elif hasattr(value, '__str__'):
        return str(value)
    else:
        raise ValueError(f"Unsupported type for conversion to string: {type(value)}")
