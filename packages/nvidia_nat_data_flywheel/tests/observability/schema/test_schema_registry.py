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

import logging

import pytest
from pydantic import BaseModel
from pydantic import Field

from nat.plugins.data_flywheel.observability.schema.schema_registry import SchemaRegistry
from nat.plugins.data_flywheel.observability.schema.schema_registry import get_available_destinations
from nat.plugins.data_flywheel.observability.schema.schema_registry import get_available_schemas
from nat.plugins.data_flywheel.observability.schema.schema_registry import get_schema
from nat.plugins.data_flywheel.observability.schema.schema_registry import get_schemas_for_destination
from nat.plugins.data_flywheel.observability.schema.schema_registry import is_registered
from nat.plugins.data_flywheel.observability.schema.schema_registry import register_schema


# Mock schema classes for testing
class MockSchemaV1(BaseModel):
    """Mock schema version 1.0."""
    name: str = Field(..., description="Name field")
    version: str = Field(default="1.0", description="Version field")


class MockSchemaV2(BaseModel):
    """Mock schema version 2.0."""
    name: str = Field(..., description="Name field")
    version: str = Field(default="2.0", description="Version field")
    new_field: str = Field(default="default", description="New field in v2")


class ElasticsearchSchemaV1(BaseModel):
    """Elasticsearch schema version 1.0."""
    index: str = Field(..., description="Index name")
    doc_type: str = Field(..., description="Document type")


class ElasticsearchSchemaV2(BaseModel):
    """Elasticsearch schema version 2.0."""
    index: str = Field(..., description="Index name")
    # doc_type removed in v2
    mappings: dict = Field(default_factory=dict, description="Index mappings")


class TestSchemaRegistry:
    """Test cases for SchemaRegistry class."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup registry for test isolation."""
        # Clear registry before each test
        SchemaRegistry.clear()

        yield  # Run the test

        # Clean up after each test
        SchemaRegistry.clear()

    def test_register_decorator_basic(self):
        """Test basic schema registration using decorator."""

        @SchemaRegistry.register("test", "1.0")
        class MySchema(BaseModel):
            field: str

        # Verify schema is registered
        assert SchemaRegistry.is_registered("test", "1.0")
        retrieved_schema = SchemaRegistry.get_schema("test", "1.0")
        assert retrieved_schema == MySchema
        assert retrieved_schema.__name__ == "MySchema"

    def test_register_decorator_multiple_schemas(self):
        """Test registering multiple schemas for different destinations and versions."""

        @SchemaRegistry.register("elasticsearch", "1.0")
        class ESSchemaV1(BaseModel):
            field1: str

        @SchemaRegistry.register("elasticsearch", "2.0")
        class ESSchemaV2(BaseModel):
            field1: str
            field2: str

        @SchemaRegistry.register("postgres", "1.0")
        class PGSchemaV1(BaseModel):
            table: str

        # Verify all schemas are registered correctly
        assert SchemaRegistry.is_registered("elasticsearch", "1.0")
        assert SchemaRegistry.is_registered("elasticsearch", "2.0")
        assert SchemaRegistry.is_registered("postgres", "1.0")

        # Verify correct schemas are retrieved
        assert SchemaRegistry.get_schema("elasticsearch", "1.0") == ESSchemaV1
        assert SchemaRegistry.get_schema("elasticsearch", "2.0") == ESSchemaV2
        assert SchemaRegistry.get_schema("postgres", "1.0") == PGSchemaV1

    def test_register_decorator_override_warning(self, caplog):
        """Test that overriding existing schema logs a warning."""

        @SchemaRegistry.register("test", "1.0")
        class OriginalSchema(BaseModel):
            field: str

        # Register another schema with the same name:version
        with caplog.at_level(logging.WARNING):

            @SchemaRegistry.register("test", "1.0")
            class NewSchema(BaseModel):
                field: str
                new_field: str

        # Check warning was logged
        assert "Overriding existing schema for test:1.0" in caplog.text

        # Verify the new schema replaced the old one
        retrieved = SchemaRegistry.get_schema("test", "1.0")
        assert retrieved == NewSchema

    def test_register_decorator_debug_logging(self, caplog):
        """Test that schema registration logs debug message."""
        with caplog.at_level(logging.DEBUG):

            @SchemaRegistry.register("test", "1.0")
            class TestSchema(BaseModel):
                field: str

        assert "Registered schema TestSchema for test:1.0" in caplog.text

    def test_get_schema_valid_cases(self):
        """Test getting schemas for valid name:version combinations."""
        # Register test schemas
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "2.0")(MockSchemaV2)
        SchemaRegistry.register("elasticsearch", "1.0")(ElasticsearchSchemaV1)

        # Test retrieval
        schema_v1 = SchemaRegistry.get_schema("test", "1.0")
        schema_v2 = SchemaRegistry.get_schema("test", "2.0")
        es_schema = SchemaRegistry.get_schema("elasticsearch", "1.0")

        assert schema_v1 == MockSchemaV1
        assert schema_v2 == MockSchemaV2
        assert es_schema == ElasticsearchSchemaV1

    def test_get_schema_destination_not_found(self):
        """Test KeyError when destination is not registered."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)

        with pytest.raises(KeyError, match="Destination 'nonexistent' not found"):
            SchemaRegistry.get_schema("nonexistent", "1.0")

    def test_get_schema_version_not_found(self):
        """Test KeyError when version is not registered for existing destination."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)

        with pytest.raises(KeyError, match="Version '2.0' not found for destination 'test'"):
            SchemaRegistry.get_schema("test", "2.0")

    def test_get_schema_error_messages_include_available_options(self):
        """Test that error messages include available destinations/versions."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "1.1")(MockSchemaV2)
        SchemaRegistry.register("elasticsearch", "1.0")(ElasticsearchSchemaV1)

        # Test destination not found includes available destinations
        with pytest.raises(KeyError) as exc_info:
            SchemaRegistry.get_schema("nonexistent", "1.0")
        error_message = str(exc_info.value)
        assert "Available destinations: ['test', 'elasticsearch']" in error_message

        # Test version not found includes available versions
        with pytest.raises(KeyError) as exc_info:
            SchemaRegistry.get_schema("test", "2.0")
        error_message = str(exc_info.value)
        assert "Available versions: ['1.0', '1.1']" in error_message

    def test_get_available_schemas_empty_registry(self):
        """Test get_available_schemas with empty registry."""
        schemas = SchemaRegistry.get_available_schemas()
        assert schemas == []

    def test_get_available_schemas_multiple_schemas(self):
        """Test get_available_schemas with multiple registered schemas."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "2.0")(MockSchemaV2)
        SchemaRegistry.register("elasticsearch", "1.0")(ElasticsearchSchemaV1)
        SchemaRegistry.register("elasticsearch", "2.0")(ElasticsearchSchemaV2)

        schemas = SchemaRegistry.get_available_schemas()
        expected = ["test:1.0", "test:2.0", "elasticsearch:1.0", "elasticsearch:2.0"]

        # Sort both lists since order may vary
        assert sorted(schemas) == sorted(expected)

    def test_get_schemas_for_destination_existing(self):
        """Test get_schemas_for_destination for existing destination."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "1.1")(MockSchemaV2)
        SchemaRegistry.register("test", "2.0")(MockSchemaV2)

        versions = SchemaRegistry.get_schemas_for_destination("test")
        expected = ["1.0", "1.1", "2.0"]
        assert sorted(versions) == sorted(expected)

    def test_get_schemas_for_destination_nonexistent(self):
        """Test get_schemas_for_destination for nonexistent destination."""
        versions = SchemaRegistry.get_schemas_for_destination("nonexistent")
        assert versions == []

    def test_get_available_destinations_empty_registry(self):
        """Test get_available_destinations with empty registry."""
        destinations = SchemaRegistry.get_available_destinations()
        assert destinations == []

    def test_get_available_destinations_multiple_destinations(self):
        """Test get_available_destinations with multiple destinations."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("elasticsearch", "1.0")(ElasticsearchSchemaV1)
        SchemaRegistry.register("postgres", "1.0")(MockSchemaV1)

        destinations = SchemaRegistry.get_available_destinations()
        expected = ["test", "elasticsearch", "postgres"]
        assert sorted(destinations) == sorted(expected)

    def test_is_registered_true_cases(self):
        """Test is_registered returns True for registered schemas."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("elasticsearch", "2.0")(ElasticsearchSchemaV2)

        assert SchemaRegistry.is_registered("test", "1.0") is True
        assert SchemaRegistry.is_registered("elasticsearch", "2.0") is True

    def test_is_registered_false_cases(self):
        """Test is_registered returns False for unregistered schemas."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)

        # Destination doesn't exist
        assert SchemaRegistry.is_registered("nonexistent", "1.0") is False

        # Destination exists but version doesn't
        assert SchemaRegistry.is_registered("test", "2.0") is False

        # Neither exists
        assert SchemaRegistry.is_registered("other", "3.0") is False

    def test_clear_registry(self):
        """Test clearing the registry removes all schemas."""
        # Register some schemas
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "2.0")(MockSchemaV2)
        SchemaRegistry.register("elasticsearch", "1.0")(ElasticsearchSchemaV1)

        # Verify schemas are registered
        assert len(SchemaRegistry.get_available_schemas()) == 3
        assert SchemaRegistry.is_registered("test", "1.0")

        # Clear registry
        SchemaRegistry.clear()

        # Verify registry is empty
        assert len(SchemaRegistry.get_available_schemas()) == 0
        assert len(SchemaRegistry.get_available_destinations()) == 0
        assert not SchemaRegistry.is_registered("test", "1.0")

    def test_convenience_aliases_register_schema(self):
        """Test register_schema convenience alias works correctly."""

        @register_schema("test", "1.0")
        class TestSchema(BaseModel):
            field: str

        assert SchemaRegistry.is_registered("test", "1.0")
        assert SchemaRegistry.get_schema("test", "1.0") == TestSchema

    def test_convenience_aliases_get_schema(self):
        """Test get_schema convenience alias works correctly."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)

        # Use convenience alias
        schema = get_schema("test", "1.0")
        assert schema == MockSchemaV1

    def test_convenience_aliases_get_available_schemas(self):
        """Test get_available_schemas convenience alias works correctly."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "2.0")(MockSchemaV2)

        # Use convenience alias
        schemas = get_available_schemas()
        expected = ["test:1.0", "test:2.0"]
        assert sorted(schemas) == sorted(expected)

    def test_convenience_aliases_get_available_destinations(self):
        """Test get_available_destinations convenience alias works correctly."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("elasticsearch", "1.0")(ElasticsearchSchemaV1)

        # Use convenience alias
        destinations = get_available_destinations()
        expected = ["test", "elasticsearch"]
        assert sorted(destinations) == sorted(expected)

    def test_convenience_aliases_get_schemas_for_destination(self):
        """Test get_schemas_for_destination convenience alias works correctly."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)
        SchemaRegistry.register("test", "2.0")(MockSchemaV2)

        # Use convenience alias
        versions = get_schemas_for_destination("test")
        expected = ["1.0", "2.0"]
        assert sorted(versions) == sorted(expected)

    def test_convenience_aliases_is_registered(self):
        """Test is_registered convenience alias works correctly."""
        SchemaRegistry.register("test", "1.0")(MockSchemaV1)

        # Use convenience alias
        assert is_registered("test", "1.0") is True
        assert is_registered("test", "2.0") is False

    def test_registry_state_isolation(self):
        """Test that registry state is properly isolated between operations."""
        # Register first schema
        SchemaRegistry.register("test1", "1.0")(MockSchemaV1)
        assert len(SchemaRegistry.get_available_schemas()) == 1

        # Register second schema
        SchemaRegistry.register("test2", "1.0")(MockSchemaV2)
        assert len(SchemaRegistry.get_available_schemas()) == 2

        # Verify both are accessible
        assert SchemaRegistry.get_schema("test1", "1.0") == MockSchemaV1
        assert SchemaRegistry.get_schema("test2", "1.0") == MockSchemaV2

    def test_schema_class_preservation(self):
        """Test that registered schema classes preserve their properties."""

        @SchemaRegistry.register("test", "1.0")
        class PreservationTest(BaseModel):
            """Test docstring preservation."""
            field1: str = Field(..., description="Field 1")
            field2: int = Field(default=42, description="Field 2")

            def custom_method(self):
                return "custom"

        # Retrieve schema and verify properties are preserved
        retrieved = SchemaRegistry.get_schema("test", "1.0")

        assert retrieved.__name__ == "PreservationTest"
        assert retrieved.__doc__ == "Test docstring preservation."

        # Test that we can create instances
        instance = retrieved(field1="test")
        assert instance.field1 == "test"
        assert instance.field2 == 42
        assert instance.custom_method() == "custom"

    def test_complex_version_strings(self):
        """Test registration and retrieval with complex version strings."""
        complex_versions = ["1.0.0", "1.0.0-alpha", "1.0.0-beta.1", "2.0.0-rc.1"]

        for version in complex_versions:
            SchemaRegistry.register("test", version)(MockSchemaV1)

        # Verify all versions are registered
        for version in complex_versions:
            assert SchemaRegistry.is_registered("test", version)
            assert SchemaRegistry.get_schema("test", version) == MockSchemaV1

        # Verify all versions are listed
        versions = SchemaRegistry.get_schemas_for_destination("test")
        assert sorted(versions) == sorted(complex_versions)

    def test_unicode_destination_names(self):
        """Test registration with unicode destination names."""
        unicode_destinations = ["测试", "тест", "🚀_destination"]

        for dest in unicode_destinations:
            SchemaRegistry.register(dest, "1.0")(MockSchemaV1)

        # Verify all unicode destinations work
        for dest in unicode_destinations:
            assert SchemaRegistry.is_registered(dest, "1.0")
            assert SchemaRegistry.get_schema(dest, "1.0") == MockSchemaV1

        destinations = SchemaRegistry.get_available_destinations()
        assert sorted(destinations) == sorted(unicode_destinations)
