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


def deserialize_input_value(input_value: dict[str, Any] | list[Any] | str) -> dict[str, Any] | list[Any]:
    """Deserialize a string input value to a dictionary, list, or None.

    Args:
        input_value (str): The input value to deserialize

    Returns:
        dict | list: The deserialized input value

    Raises:
        ValueError: If parsing fails
    """
    try:
        if isinstance(input_value, (dict, list)):
            return input_value
        deserialized_attribute = json.loads(input_value)
        return deserialized_attribute
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Failed to parse input_value: {input_value}, error: {e}") from e
