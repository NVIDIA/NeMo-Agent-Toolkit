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

from nat.plugins.mcp.utils import model_from_mcp_schema


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
    assert "optional_string_field_no_default" in _model.model_fields
    assert m.optional_string_field_no_default is None
    field_type = _model.model_fields["optional_string_field_no_default"].annotation
    args = get_args(field_type)
    assert str in args and type(None) in args, f"Expected str | None, got {field_type}"

    # Check that the optional union field is present
    assert "optional_union_field" in _model.model_fields
    assert m.optional_union_field is None
    field_type = _model.model_fields["optional_union_field"].annotation
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


def test_anyof_array_and_null():
    """Test that anyOf with array and null is correctly handled"""
    schema = {
        'type': 'object',
        'properties': {
            'output_fields': {
                'description': 'Fields to output',
                'anyOf': [{
                    'type': 'array', 'items': {
                        'type': 'string'
                    }
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_anyof_model", schema)

    # Verify the field exists
    assert "output_fields" in _model.model_fields

    # Verify the field type is list[str] | None
    field_type = _model.model_fields["output_fields"].annotation
    args = get_args(field_type)

    # Check that we have list and NoneType in the union
    assert type(None) in args, f"Expected None in union, got {field_type}"

    # Find the list type
    list_types = [arg for arg in args if hasattr(arg, '__origin__') and arg.__origin__ is list]
    assert len(list_types) > 0, f"Expected list type in union, got {field_type}"

    # Check that list contains str
    list_type = list_types[0]
    list_args = get_args(list_type)
    assert str in list_args, f"Expected list[str], got {list_type}"

    # Test with array value
    m1 = _model.model_validate({"output_fields": ["field1", "field2"]})
    assert m1.output_fields == ["field1", "field2"]

    # Test with null value
    m2 = _model.model_validate({"output_fields": None})
    assert m2.output_fields is None

    # Test with missing field (should default to None)
    m3 = _model.model_validate({})
    assert m3.output_fields is None


def test_anyof_string_and_number():
    """Test that anyOf with string and number is correctly handled"""
    schema = {
        'type': 'object',
        'properties': {
            'flexible_field': {
                'description': 'A field that can be string or number',
                'anyOf': [{
                    'type': 'string'
                }, {
                    'type': 'number'
                }]
            }
        },
        'required': ['flexible_field']
    }

    _model = model_from_mcp_schema("test_anyof_string_number", schema)

    # Verify the field type is str | float
    field_type = _model.model_fields["flexible_field"].annotation
    args = get_args(field_type)
    assert str in args and float in args, f"Expected str | float, got {field_type}"

    # Test with string
    m1 = _model.model_validate({"flexible_field": "test"})
    assert m1.flexible_field == "test"

    # Test with number
    m2 = _model.model_validate({"flexible_field": 42.5})
    assert m2.flexible_field == 42.5


def test_oneof_union():
    """Test that oneOf is correctly handled (similar to anyOf)"""
    schema = {
        'type': 'object',
        'properties': {
            'oneof_field': {
                'description': 'A field with oneOf', 'oneOf': [{
                    'type': 'integer'
                }, {
                    'type': 'boolean'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_oneof_model", schema)

    # Verify the field type is int | bool
    field_type = _model.model_fields["oneof_field"].annotation
    args = get_args(field_type)
    assert int in args and bool in args, f"Expected int | bool, got {field_type}"

    # Test with integer
    m1 = _model.model_validate({"oneof_field": 42})
    assert m1.oneof_field == 42

    # Test with boolean
    m2 = _model.model_validate({"oneof_field": True})
    assert m2.oneof_field is True


def test_anyof_with_object():
    """Test that anyOf with object types is correctly handled"""
    schema = {
        'type': 'object',
        'properties': {
            'config': {
                'description': 'Configuration object or null',
                'anyOf': [{
                    'type': 'object', 'properties': {
                        'setting': {
                            'type': 'string'
                        }
                    }
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_anyof_object", schema)

    # Verify the field exists
    assert "config" in _model.model_fields

    # Test with object value
    m1 = _model.model_validate({"config": {"setting": "value"}})
    assert m1.config.setting == "value"

    # Test with null value
    m2 = _model.model_validate({"config": None})
    assert m2.config is None


def test_anyof_required_field():
    """Test that anyOf works correctly with required fields"""
    schema = {
        'type': 'object',
        'properties': {
            'required_union': {
                'description': 'A required field with anyOf', 'anyOf': [{
                    'type': 'string'
                }, {
                    'type': 'integer'
                }]
            }
        },
        'required': ['required_union']
    }

    _model = model_from_mcp_schema("test_anyof_required", schema)

    # Should be able to create with string
    m1 = _model.model_validate({"required_union": "test"})
    assert m1.required_union == "test"

    # Should be able to create with integer
    m2 = _model.model_validate({"required_union": 42})
    assert m2.required_union == 42

    # Should fail without the field
    with pytest.raises(ValidationError):
        _model.model_validate({})


def test_anyof_array_of_objects():
    """Test that anyOf with array of objects is correctly handled"""
    schema = {
        'type': 'object',
        'properties': {
            'items': {
                'description':
                    'Array of items or null',
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'object', 'properties': {
                            'id': {
                                'type': 'integer'
                            }, 'name': {
                                'type': 'string'
                            }
                        }
                    }
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_anyof_array_objects", schema)

    # Test with array of objects
    m1 = _model.model_validate({"items": [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]})
    assert len(m1.items) == 2
    assert m1.items[0].id == 1
    assert m1.items[0].name == "Item 1"

    # Test with null
    m2 = _model.model_validate({"items": None})
    assert m2.items is None


def test_nested_anyof_in_array_items():
    """Test that anyOf within array items is correctly handled"""
    schema = {
        'type': 'object',
        'properties': {
            'mixed_array': {
                'description': 'Array with items that can be string or integer',
                'type': 'array',
                'items': {
                    'anyOf': [{
                        'type': 'string'
                    }, {
                        'type': 'integer'
                    }]
                }
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_nested_anyof_array", schema)

    # Verify the field exists
    assert "mixed_array" in _model.model_fields

    # Test with mixed array
    m1 = _model.model_validate({"mixed_array": ["hello", 42, "world", 100]})
    assert len(m1.mixed_array) == 4
    assert m1.mixed_array[0] == "hello"
    assert m1.mixed_array[1] == 42

    # Test with missing field (should default to None)
    m2 = _model.model_validate({})
    assert m2.mixed_array is None


def test_nested_anyof_in_object_properties():
    """Test that anyOf within object properties is correctly handled"""
    schema = {
        'type': 'object',
        'properties': {
            'user': {
                'description': 'User object with flexible fields',
                'type': 'object',
                'properties': {
                    'id': {
                        'type': 'integer'
                    }, 'age_or_name': {
                        'anyOf': [{
                            'type': 'integer'
                        }, {
                            'type': 'string'
                        }]
                    }
                }
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_nested_anyof_object", schema)

    # Test with integer value
    m1 = _model.model_validate({"user": {"id": 1, "age_or_name": 25}})
    assert m1.user.id == 1
    assert m1.user.age_or_name == 25

    # Test with string value
    m2 = _model.model_validate({"user": {"id": 2, "age_or_name": "John"}})
    assert m2.user.id == 2
    assert m2.user.age_or_name == "John"


def test_anyof_array_with_anyof_items():
    """Test anyOf containing an array whose items also have anyOf"""
    schema = {
        'type': 'object',
        'properties': {
            'flexible_data': {
                'description':
                    'Either an array of mixed types or null',
                'anyOf': [{
                    'type': 'array', 'items': {
                        'anyOf': [{
                            'type': 'string'
                        }, {
                            'type': 'number'
                        }]
                    }
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_double_anyof", schema)

    # Test with array of mixed types
    m1 = _model.model_validate({"flexible_data": ["hello", 3.14, "world", 42]})
    assert len(m1.flexible_data) == 4
    assert m1.flexible_data[0] == "hello"
    assert m1.flexible_data[1] == 3.14

    # Test with null
    m2 = _model.model_validate({"flexible_data": None})
    assert m2.flexible_data is None


def test_oneof_with_nested_object():
    """Test oneOf containing objects with nested anyOf"""
    schema = {
        'type': 'object',
        'properties': {
            'config': {
                'description':
                    'Configuration with multiple formats',
                'oneOf': [{
                    'type': 'object',
                    'properties': {
                        'mode': {
                            'type': 'string'
                        }, 'value': {
                            'anyOf': [{
                                'type': 'integer'
                            }, {
                                'type': 'boolean'
                            }]
                        }
                    }
                }, {
                    'type': 'string'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_oneof_nested", schema)

    # Test with object containing anyOf field
    m1 = _model.model_validate({"config": {"mode": "auto", "value": 42}})
    assert m1.config.mode == "auto"
    assert m1.config.value == 42

    # Test with object containing boolean in anyOf field
    m2 = _model.model_validate({"config": {"mode": "manual", "value": True}})
    assert m2.config.mode == "manual"
    assert m2.config.value is True

    # Test with string alternative
    m3 = _model.model_validate({"config": "default"})
    assert m3.config == "default"


def test_deeply_nested_anyof():
    """Test deeply nested anyOf structures"""
    schema = {
        'type': 'object',
        'properties': {
            'data': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'nested_field': {
                                'anyOf': [{
                                    'type': 'string'
                                }, {
                                    'type': 'null'
                                }]
                            }
                        }
                    }
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_deeply_nested", schema)

    # Test with array of objects with anyOf fields
    m1 = _model.model_validate(
        {"data": [{
            "nested_field": "value1"
        }, {
            "nested_field": None
        }, {
            "nested_field": "value2"
        }]})
    assert len(m1.data) == 3
    assert m1.data[0].nested_field == "value1"
    assert m1.data[1].nested_field is None
    assert m1.data[2].nested_field == "value2"

    # Test with null
    m2 = _model.model_validate({"data": None})
    assert m2.data is None


def test_anyof_with_array_of_objects_with_anyof():
    """Test anyOf at top level with array items that contain objects with anyOf properties"""
    schema = {
        'type': 'object',
        'properties': {
            'results': {
                'description':
                    'Results can be an array of items or null',
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'id': {
                                'type': 'integer'
                            },
                            'status': {
                                'anyOf': [{
                                    'type': 'string'
                                }, {
                                    'type': 'integer'
                                }, {
                                    'type': 'null'
                                }]
                            }
                        }
                    }
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_complex_nested", schema)

    # Test with array of objects with various status types
    m1 = _model.model_validate(
        {"results": [{
            "id": 1, "status": "active"
        }, {
            "id": 2, "status": 200
        }, {
            "id": 3, "status": None
        }]})
    assert len(m1.results) == 3
    assert m1.results[0].status == "active"
    assert m1.results[1].status == 200
    assert m1.results[2].status is None

    # Test with null
    m2 = _model.model_validate({"results": None})
    assert m2.results is None


def test_required_nullable_field_with_anyof():
    """Test that required nullable fields enforce presence but accept None as a value"""
    schema = {
        'type': 'object',
        'properties': {
            'nullable_field': {
                'description': 'Required field that can be null', 'anyOf': [{
                    'type': 'string'
                }, {
                    'type': 'null'
                }]
            }
        },
        'required': ['nullable_field']
    }

    _model = model_from_mcp_schema("test_required_nullable", schema)

    # Verify field type is str | None
    field_type = _model.model_fields["nullable_field"].annotation
    args = get_args(field_type)
    assert str in args and type(None) in args, f"Expected str | None, got {field_type}"

    # Test with string value - should succeed
    m1 = _model.model_validate({"nullable_field": "test"})
    assert m1.nullable_field == "test"

    # Test with None value - should succeed
    m2 = _model.model_validate({"nullable_field": None})
    assert m2.nullable_field is None

    # Test with missing field - should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        _model.model_validate({})
    errors = exc_info.value.errors()
    missing_fields = {e['loc'][0] for e in errors if e['type'] == 'missing'}
    assert 'nullable_field' in missing_fields


def test_required_nullable_field_with_type_list():
    """Test required nullable field using type list notation"""
    schema = {
        'type': 'object',
        'properties': {
            'nullable_int': {
                'description': 'Required int or null', 'type': ['integer', 'null']
            }
        },
        'required': ['nullable_int']
    }

    _model = model_from_mcp_schema("test_required_nullable_list", schema)

    # Verify field type is int | None
    field_type = _model.model_fields["nullable_int"].annotation
    args = get_args(field_type)
    assert int in args and type(None) in args, f"Expected int | None, got {field_type}"

    # Test with integer value - should succeed
    m1 = _model.model_validate({"nullable_int": 42})
    assert m1.nullable_int == 42

    # Test with None value - should succeed
    m2 = _model.model_validate({"nullable_int": None})
    assert m2.nullable_int is None

    # Test with missing field - should raise ValidationError
    with pytest.raises(ValidationError) as exc_info:
        _model.model_validate({})
    errors = exc_info.value.errors()
    missing_fields = {e['loc'][0] for e in errors if e['type'] == 'missing'}
    assert 'nullable_int' in missing_fields


def test_required_nullable_field_with_enum():
    """Test that enum containing null is detected correctly for required fields"""
    schema = {
        'type': 'object',
        'properties': {
            'enum_field': {
                'description': 'Required field with enum including null', 'enum': ['value1', 'value2', None]
            }
        },
        'required': ['enum_field']
    }

    _model = model_from_mcp_schema("test_required_nullable_enum", schema)

    # Field should be required (missing field should raise error)
    with pytest.raises(ValidationError) as exc_info:
        _model.model_validate({})
    errors = exc_info.value.errors()
    missing_fields = {e['loc'][0] for e in errors if e['type'] == 'missing'}
    assert 'enum_field' in missing_fields

    # But should accept None as a valid value
    m1 = _model.model_validate({"enum_field": None})
    assert m1.enum_field is None


def test_required_nullable_field_with_const_null():
    """Test that const: null is detected correctly for required fields"""
    schema = {
        'type': 'object',
        'properties': {
            'const_null_field': {
                'description': 'Required field with const null', 'const': None
            }
        },
        'required': ['const_null_field']
    }

    _model = model_from_mcp_schema("test_required_const_null", schema)

    # Field should be required (missing field should raise error)
    with pytest.raises(ValidationError) as exc_info:
        _model.model_validate({})
    errors = exc_info.value.errors()
    missing_fields = {e['loc'][0] for e in errors if e['type'] == 'missing'}
    assert 'const_null_field' in missing_fields

    # Should accept None as value
    m1 = _model.model_validate({"const_null_field": None})
    assert m1.const_null_field is None


def test_type_list_with_array_items():
    """Test that type list containing 'array' properly resolves item types"""
    schema = {
        'type': 'object',
        'properties': {
            'mixed_field': {
                'description': 'Field that can be array of strings or null',
                'type': ['array', 'null'],
                'items': {
                    'type': 'string'
                }
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_type_list_array", schema)

    # Verify field type includes list[str] and None
    field_type = _model.model_fields["mixed_field"].annotation
    args = get_args(field_type)
    assert type(None) in args, f"Expected None in union, got {field_type}"

    # Find the list type
    list_types = [arg for arg in args if hasattr(arg, '__origin__') and arg.__origin__ is list]
    assert len(list_types) > 0, f"Expected list type in union, got {field_type}"

    # Check that list contains str
    list_type = list_types[0]
    list_args = get_args(list_type)
    assert str in list_args, f"Expected list[str], got {list_type}"

    # Test with array value
    m1 = _model.model_validate({"mixed_field": ["a", "b", "c"]})
    assert m1.mixed_field == ["a", "b", "c"]

    # Test with null value
    m2 = _model.model_validate({"mixed_field": None})
    assert m2.mixed_field is None


def test_type_list_with_object_properties():
    """Test that type list containing 'object' properly resolves property types"""
    schema = {
        'type': 'object',
        'properties': {
            'config_or_null': {
                'description': 'Config object or null',
                'type': ['object', 'null'],
                'properties': {
                    'setting': {
                        'type': 'string'
                    }, 'value': {
                        'type': 'integer'
                    }
                }
            }
        },
        'required': []
    }

    _model = model_from_mcp_schema("test_type_list_object", schema)

    # Test with object value
    m1 = _model.model_validate({"config_or_null": {"setting": "test", "value": 123}})
    assert m1.config_or_null.setting == "test"
    assert m1.config_or_null.value == 123

    # Test with null value
    m2 = _model.model_validate({"config_or_null": None})
    assert m2.config_or_null is None
