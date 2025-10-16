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

import yaml
from pydantic import ValidationError

from ..exception_handlers.schemas import schema_exception_handler
from ..exception_handlers.schemas import yaml_exception_handler


@schema_exception_handler
def validate_schema(metadata, Schema):

    try:
        return Schema(**metadata)
    except ValidationError as e:

        raise e


@yaml_exception_handler
def validate_yaml(ctx, param, value):
    """
    Validate that the file is a valid YAML file

    Parameters
    ----------
    ctx: Click context
    param: Click parameter
    value: Path to YAML file

    Returns
    -------
    str: Path to valid YAML file

    Raises
    ------
    ValueError: If file is invalid or unreadable
    """
    if value is None:
        return None

    with open(value, encoding="utf-8") as f:
        yaml.safe_load(f)

    return value
