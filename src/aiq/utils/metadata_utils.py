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

from pydantic_core import PydanticUndefined

from aiq.data_models.common import TypedBaseModelT
from aiq.utils.type_utils import DecomposedType


def generate_config_type_docs(config_type: TypedBaseModelT) -> str:
    """Generates a docstring from configuration object to facilitate discovery.

    Args:
        config_type (TypedBaseModelT): A component configuration object.

    Returns:
        str: An enriched docstring, including model attributes and default values.
    """

    # Get the docstring
    description_formatting = []
    # Ensure uniform formatting of docstring
    docstring = (config_type.__doc__ or "").strip().strip(".")
    docstring = docstring + "." if docstring != "" else "Description unavailable."
    description_formatting.append(docstring)
    description_formatting.append("")
    description_formatting.append("  Args:")

    # Iterate over fields to get their documentation
    for field_name, field_info in config_type.model_fields.items():

        if (field_name == "type"):
            field_name = "_type"

        decomponsed_type = DecomposedType(field_info.annotation)

        if not (decomponsed_type.is_union):
            annotation = field_info.annotation.__name__
        else:
            annotation = field_info.annotation

        default_string = ""
        if ((field_info.get_default() is not PydanticUndefined) and (field_name != "_type")):
            if issubclass(type(field_info.get_default()), str):
                default_value = f'"{field_info.get_default()}"'
            else:
                default_value = field_info.get_default()
            default_string += f" Defaults to {default_value}."

        # Ensure uniform formatting of field info
        field_info_description = (field_info.description or "").strip(".")
        if field_info_description != "":
            field_info_description = field_info_description + "."
        else:
            field_info_description = "Description unavailable."

        parameter_string = f"    {field_name} ({annotation}): {field_info_description}{default_string}"
        description_formatting.append(parameter_string)

    description = "\n".join(description_formatting)

    return description
