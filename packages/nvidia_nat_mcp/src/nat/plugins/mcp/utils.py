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

from enum import Enum
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import create_model


def truncate_session_id(session_id: str, max_length: int = 10) -> str:
    """
    Truncate a session ID for logging purposes.

    Args:
        session_id: The session ID to truncate
        max_length: Maximum length before truncation (default: 10)

    Returns:
        Truncated session ID with "..." if longer than max_length, otherwise full ID
    """
    if len(session_id) > max_length:
        return session_id[:max_length] + "..."
    return session_id


def model_from_mcp_schema(name: str, mcp_input_schema: dict) -> type[BaseModel]:
    """
    Create a pydantic model from the input schema of the MCP tool
    """
    _type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "null": type(None),
        "object": dict,
    }

    properties = mcp_input_schema.get("properties", {})
    required_fields = set(mcp_input_schema.get("required", []))
    schema_dict = {}

    def _generate_valid_classname(class_name: str):
        return class_name.replace('_', ' ').replace('-', ' ').title().replace(' ', '')

    def _resolve_schema_type(schema: dict[str, Any], name: str) -> Any:
        """
        Recursively resolve a JSON schema to a Python type.
        Handles nested anyOf/oneOf, arrays, objects, enums, and primitive types.
        """
        # Check for anyOf/oneOf first
        any_of = schema.get("anyOf")
        one_of = schema.get("oneOf")

        if any_of or one_of:
            union_schemas = any_of if any_of else one_of
            resolved_type: Any = None

            if union_schemas:
                for sub_schema in union_schemas:
                    mapped = _resolve_schema_type(sub_schema, name)
                    if resolved_type is None:
                        resolved_type = mapped
                    elif mapped is not type(None):
                        # Don't add None here, handle separately
                        resolved_type = resolved_type | mapped
                    else:
                        # If we encounter null, combine with None at the end
                        resolved_type = resolved_type | None if resolved_type else type(None)

            return resolved_type if resolved_type is not None else Any

        # Handle enum values
        enum_vals = schema.get("enum")
        if enum_vals:
            # Check if enum contains null
            has_null = any(val is None or val == "null" for val in enum_vals)
            # Filter out None/null values from enum
            non_null_vals = [v for v in enum_vals if v is not None and v != "null"]

            if non_null_vals:
                enum_name = f"{name.capitalize()}Enum"
                enum_type: Any = Enum(enum_name, {item: item for item in non_null_vals})
                # If enum had null, make it a union with None
                return enum_type | None if has_null else enum_type
            elif has_null:
                # Enum only contains null
                return type(None)
            else:
                # Empty enum (shouldn't happen but handle gracefully)
                return Any

        schema_type = schema.get("type")

        # Handle type as list (e.g., ["string", "integer", "null"])
        if isinstance(schema_type, list):
            list_type: Any = None
            for t in schema_type:
                if t == "array":
                    # Incorporate the mapped type of items
                    item_schema = schema.get("items", {})
                    if item_schema:
                        item_type = _resolve_schema_type(item_schema, name)
                        mapped = list[item_type]
                    else:
                        mapped = _type_map.get(t, Any)
                elif t == "object":
                    # Incorporate the mapped type from properties
                    if "properties" in schema:
                        mapped = model_from_mcp_schema(name=name, mcp_input_schema=schema)
                    else:
                        mapped = _type_map.get(t, Any)
                else:
                    mapped = _type_map.get(t, Any)

                list_type = mapped if list_type is None else list_type | mapped
            return list_type if list_type is not None else Any

        # Handle null type
        if schema_type == "null":
            return type(None)

        # Handle object type
        if schema_type == "object" and "properties" in schema:
            return model_from_mcp_schema(name=name, mcp_input_schema=schema)

        # Handle array type
        if schema_type == "array" and "items" in schema:
            item_schema = schema.get("items", {})
            # Recursively resolve item type (handles nested anyOf/oneOf)
            item_type = _resolve_schema_type(item_schema, name)
            return list[item_type]

        # Handle primitive types
        if schema_type is not None:
            return _type_map.get(schema_type, Any)

        return Any

    def _has_null_in_type(field_properties: dict[str, Any]) -> bool:
        """Check if a schema contains null as a valid type."""
        # Check anyOf/oneOf for null
        any_of = field_properties.get("anyOf")
        one_of = field_properties.get("oneOf")
        if any_of or one_of:
            union_schemas = any_of if any_of else one_of
            if union_schemas:
                for schema in union_schemas:
                    if schema.get("type") == "null":
                        return True

        # Check type list for null
        json_type = field_properties.get("type")
        if isinstance(json_type, list) and "null" in json_type:
            return True

        # Check enum for null (Python None or string "null")
        enum_vals = field_properties.get("enum")
        if enum_vals:
            for val in enum_vals:
                if val is None or val == "null":
                    return True

        # Check const for null (Python None or string "null")
        if "const" in field_properties:
            const_val = field_properties.get("const")
            if const_val is None or const_val == "null":
                return True

        return False

    def _generate_field(field_name: str, field_properties: dict[str, Any]) -> tuple:
        """
        Generate a Pydantic field from JSON schema properties.
        Uses _resolve_schema_type for type resolution and handles field-specific logic.
        """
        # Resolve the field type using the unified resolver
        field_type = _resolve_schema_type(field_properties, field_name)

        # Check if the type includes null
        has_null = _has_null_in_type(field_properties)

        # Determine the default value based on whether the field is required
        default_value = field_properties.get("default")

        if field_name in required_fields:
            # Field is required - use explicit default if provided, otherwise use ... to enforce presence
            if default_value is None and "default" not in field_properties:
                # Required field without explicit default: always use ... even if nullable
                default_value = ...
            # Make the field type nullable if it allows null
            if has_null:
                field_type = field_type | None
        else:
            # Field is optional - use explicit default if provided, otherwise None
            if default_value is None:
                default_value = None
            # Make the type optional if no default was provided and not already nullable
            if "default" not in field_properties and not has_null:
                field_type = field_type | None

        # Handle nullable property (less common, but still supported)
        nullable = field_properties.get("nullable", False)
        if nullable and not has_null:
            field_type = field_type | None

        description = field_properties.get("description", "")

        return field_type, Field(default=default_value, description=description)

    for field_name, field_props in properties.items():
        schema_dict[field_name] = _generate_field(field_name=field_name, field_properties=field_props)
    return create_model(f"{_generate_valid_classname(name)}InputSchema", **schema_dict)
