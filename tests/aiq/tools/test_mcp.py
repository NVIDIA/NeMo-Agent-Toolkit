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

from typing import get_args

import pytest
from pydantic import ValidationError
from pytest_httpserver import HTTPServer

from aiq.tool.mcp.mcp_client import model_from_mcp_schema


@pytest.fixture(name="test_mcp_server")
def _get_test_mcp_server(httpserver: HTTPServer):
    httpserver.expect_request("/sse", )


@pytest.fixture(name="sample_schema")
def _get_sample_schema():
    return {
        'description': 'Test Tool',
        'properties': {
            'required_string_field': {
                'description': 'Required field that needs to be a string',
                'minLength': 1,
                'title': 'RequiredString',
                'type': 'string'
            },
            'optional_string_field': {
                'default': 'default_string',
                'description': 'Optional field that needs to be a string',
                'minLength': 1,
                'title': 'OptionalString',
                'type': 'string'
            },
            'optional_string_field_no_default': {
                'description': 'Optional field that needs to be a string',
                'minLength': 1,
                'title': 'OptionalString',
                'type': 'string'
            },
            'optional_union_field': {
                'description': 'Optional field that needs to be a string or an integer',
                'title': 'OptionalUnion',
                'type': ['string', 'integer', 'null']
            },
            'required_int_field': {
                'description': 'Required int field.',
                'exclusiveMaximum': 1000000,
                'exclusiveMinimum': 0,
                'title': 'Required Int',
                'type': 'integer'
            },
            'optional_int_field': {
                'default': 5000,
                'description': 'Optional Integer field.',
                'exclusiveMaximum': 1000000,
                'exclusiveMinimum': 0,
                'title': 'Optional Int',
                'type': 'integer'
            },
            'required_float_field': {
                'description': 'Optional Float Field.', 'title': 'Optional Float', 'type': 'number'
            },
            'optional_float_field': {
                'default': 5.0, 'description': 'Optional Float Field.', 'title': 'Optional Float', 'type': 'number'
            },
            'optional_bool_field': {
                'default': False, 'description': 'Optional Boolean Field.', 'title': 'Raw', 'type': 'boolean'
            },
            'optional_array_field': {
                'default': ['item'],
                'description': 'Optional Array Field.',
                'title': 'Array',
                'type': 'array',
                'items': {
                    'type': 'string'
                }
            },
            'optional_array_object_field': {
                'default': [{
                    'key': 'value'
                }],
                'description': 'Optional Array Field.',
                'title': 'Array',
                'type': 'array',
                'items': {
                    'type': 'object', 'properties': {
                        'key': {
                            'type': 'string'
                        }
                    }
                }
            }
        },
        'required': [
            'required_string_field',
            'required_int_field',
            'required_float_field',
        ],
        'title': 'Fetch',
        'type': 'object'
    }


def test_schema_generation(sample_schema):
    _model = model_from_mcp_schema("test_model", sample_schema)

    for k, _ in sample_schema["properties"].items():
        assert k in _model.model_fields.keys()

    test_input = {
        "required_string_field": "This is a string",
        "optional_string_field": "This is another string",
        "required_int_field": 4,
        "optional_int_field": 1,
        "required_float_field": 5.5,
        "optional_float_field": 3.2,
        "optional_bool_field": True,
    }

    m = _model.model_validate(test_input)
    assert isinstance(m, _model)

    test_input = {
        "required_string_field": "This is a string",
        "required_int_field": 4,
        "required_float_field": 5.5,
        "optional_array_field": ["item1"],
        "optional_array_object_field": [{
            'key': 'value1'
        }],
    }

    m = _model.model_validate(test_input)
    assert isinstance(m, _model)

    # Check that the optional field with no default is
    # 1. present
    # 2. has a default value of None
    # 3. has a type of str | None
    assert hasattr(m, "optional_string_field_no_default")
    assert m.optional_string_field_no_default is None
    field_type = m.model_fields["optional_string_field_no_default"].annotation
    args = get_args(field_type)
    assert str in args and type(None) in args, f"Expected str | None, got {field_type}"

    # Check that the optional union field is present
    assert hasattr(m, "optional_union_field")
    assert m.optional_union_field is None
    field_type = m.model_fields["optional_union_field"].annotation
    args = get_args(field_type)
    assert str in args and type(None) in args and int in args, f"Expected str | None | int, got {field_type}"


def test_schema_missing_required_fields_raises(sample_schema):
    """Ensure that the required descriptor is respected in the schema generation"""
    _model = model_from_mcp_schema("test_model", sample_schema)

    incomplete_input = {
        "required_string_field": "ok",  # 'required_int_field' is missing
        "required_float_field": 5.5
    }

    with pytest.raises(ValidationError) as exc_info:
        _model.model_validate(incomplete_input)

    errors = exc_info.value.errors()
    missing_fields = {e['loc'][0] for e in errors if e['type'] == 'missing'}
    assert 'required_int_field' in missing_fields
