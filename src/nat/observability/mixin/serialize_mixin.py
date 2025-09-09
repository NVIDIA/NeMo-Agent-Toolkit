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

from pydantic import BaseModel
from pydantic import TypeAdapter


class SerializeMixin:

    def _process_streaming_output(self, input_value: Any) -> Any:
        """
        Serialize a list of values to a JSON string.
        """
        if isinstance(input_value, BaseModel):
            return json.loads(TypeAdapter(type(input_value)).dump_json(input_value).decode('utf-8'))
        if isinstance(input_value, dict):
            return input_value
        return input_value

    def _serialize_payload(self, input_value: Any) -> tuple[str, bool]:
        """
        Serialize the input value to a string. Returns a tuple with the serialized value and a boolean indicating if the
        serialization is JSON or a string.

        Args:
            input_value (Any): The input value to serialize.

        Returns:
            tuple[str, bool]: A tuple with the serialized value and a boolean indicating if the serialization is
                JSON or a string.
        """
        try:
            if isinstance(input_value, BaseModel):
                return TypeAdapter(type(input_value)).dump_json(input_value).decode('utf-8'), True
            if isinstance(input_value, dict):
                return json.dumps(input_value), True
            if isinstance(input_value, list):
                serialized_list = []
                for value in input_value:
                    serialized_value = self._process_streaming_output(value)
                    serialized_list.append(serialized_value)
                return json.dumps(serialized_list), True
            return str(input_value), False
        except Exception:
            # Fallback to string representation if we can't serialize using pydantic
            return str(input_value), False
