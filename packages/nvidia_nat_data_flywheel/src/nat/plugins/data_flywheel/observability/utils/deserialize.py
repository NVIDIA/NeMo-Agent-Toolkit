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
from typing import Any

# Type alias for all possible JSON values
JSONValue = dict[str, Any] | list[Any] | str | int | float | bool | None


def deserialize_span_attribute(value: dict[str, Any] | list[Any] | str) -> JSONValue:
    """Deserialize a string input value to any valid JSON value.

    Args:
        value (str): The input value to deserialize

    Returns:
        JSONValue: The deserialized input value, which can be a dictionary, list,
                   string, number (int/float), boolean, or None

    Raises:
        ValueError: If parsing fails
    """
    try:
        if isinstance(value, dict | list):
            return value
        deserialized_attribute = json.loads(value)
        return deserialized_attribute
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse input_value: {value}, error: {e}") from e
