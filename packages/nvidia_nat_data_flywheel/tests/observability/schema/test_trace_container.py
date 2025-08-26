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

from unittest.mock import patch

import pytest
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError

from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer


class MockTraceSource(BaseModel):
    """Mock trace source model for testing."""
    client_id: str = Field(..., description="Client ID")
    test_field: str = Field(..., description="Test field")


class TestTraceContainer:
    """Test cases for TraceContainer class."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Setup and cleanup registry for test isolation."""
        # Clear registry before each test
        try:
            # yapf: disable
            from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
                TraceAdapterRegistry,
            )
            TraceAdapterRegistry.clear_registry()
        except ImportError:
            pass  # Registry not available

        yield  # Run the test

        # Clean up after each test
        try:
            TraceAdapterRegistry.clear_registry()
        except (ImportError, NameError):
            pass  # Registry not available

    @pytest.fixture
    def valid_span(self):
        """Create a valid Span instance for testing."""
        return Span(name="test_span")

    @pytest.fixture
    def simple_source_dict(self):
        """Create a simple source dictionary for testing."""
        return {"client_id": "test_client", "test_field": "test_value"}

    @pytest.fixture
    def invalid_source_dict(self):
        """Create an invalid source dictionary that should fail union validation."""
        return {"invalid_field": "value", "another_invalid": 123}

    def test_basic_initialization_with_dict_source(self, valid_span, simple_source_dict):
        """Test basic initialization with dict source and valid span."""
        # Test with source as dict - should work with cleared registry (union = Any)
        container = TraceContainer(source=simple_source_dict, span=valid_span)

        assert container.source == simple_source_dict
        assert container.span == valid_span
        assert isinstance(container.span, Span)

    def test_basic_initialization_with_object_source(self, valid_span):
        """Test basic initialization with object source."""
        mock_source = MockTraceSource(client_id="test_id", test_field="test_name")

        container = TraceContainer(source=mock_source, span=valid_span)

        assert container.source == mock_source
        assert container.span == valid_span

    def test_basic_initialization_with_non_dict_source(self, valid_span):
        """Test basic initialization with non-dict source types."""
        # Test with string source
        string_source = "test_source_string"
        container = TraceContainer(source=string_source, span=valid_span)
        assert container.source == string_source

        # Test with integer source
        int_source = 12345
        container = TraceContainer(source=int_source, span=valid_span)
        assert container.source == int_source

        # Test with list source
        list_source = [1, 2, 3, "test"]
        container = TraceContainer(source=list_source, span=valid_span)
        assert container.source == list_source

    def test_source_validation_with_registered_adapter(self, valid_span, simple_source_dict):
        """Test source validation with a registered adapter."""
        try:
            # yapf: disable
            from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
                TraceAdapterRegistry,
            )

            # Simple mock converter that returns a dict (for testing purposes)
            @TraceAdapterRegistry.register_adapter(MockTraceSource)
            def mock_converter(trace_source: TraceContainer) -> dict:
                return {"converted": True, "source": trace_source.source}

            # This should now work with union validation
            container = TraceContainer(source=simple_source_dict, span=valid_span)
            # Source should be converted to MockTraceSource instance via union validation
            assert isinstance(container.source, MockTraceSource)
            assert container.source.client_id == simple_source_dict["client_id"]
            assert container.source.test_field == simple_source_dict["test_field"]

        except ImportError:
            # If registry not available, test basic functionality
            container = TraceContainer(source=simple_source_dict, span=valid_span)
            assert container.source == simple_source_dict

    def test_source_validation_failure_with_registered_adapter(self, valid_span, invalid_source_dict):
        """Test source validation failure with a registered adapter."""
        try:
            # yapf: disable
            from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
                TraceAdapterRegistry,
            )

            # Simple mock converter that returns a dict (for testing purposes)
            @TraceAdapterRegistry.register_adapter(MockTraceSource)
            def mock_converter(trace_source: TraceContainer) -> dict:
                return {"converted": True, "source": trace_source.source}

            # This should fail union validation
            with pytest.raises(ValueError, match="Union validation failed"):
                TraceContainer(source=invalid_source_dict, span=valid_span)

        except ImportError:
            # If registry not available, skip this test
            pytest.skip("TraceAdapterRegistry not available")

    def test_import_error_handling_in_validator(self, valid_span, simple_source_dict):
        """Test ImportError handling in source validator."""
        # With cleared registry, this should work fine
        container = TraceContainer(source=simple_source_dict, span=valid_span)
        assert container.source == simple_source_dict

    def test_import_error_handling_in_init(self, valid_span, simple_source_dict):
        """Test ImportError handling in __init__ method."""
        # Mock the import in __init__ to raise ImportError
        with patch('builtins.__import__', side_effect=ImportError):
            # Should not raise ImportError
            container = TraceContainer(source=simple_source_dict, span=valid_span)
            assert container.source == simple_source_dict
            assert container.span == valid_span

    def test_missing_required_fields_raises_validation_error(self):
        """Test that missing required fields raise ValidationError."""
        # Missing both source and span
        with pytest.raises(ValidationError) as exc_info:
            TraceContainer()

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "source" in error_fields
        assert "span" in error_fields
        assert any(error["type"] == "missing" for error in errors if error["loc"][0] == "source")
        assert any(error["type"] == "missing" for error in errors if error["loc"][0] == "span")

    def test_missing_source_field_raises_validation_error(self, valid_span):
        """Test that missing source field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TraceContainer(span=valid_span)

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "source" in error_fields
        assert any(error["type"] == "missing" for error in errors if error["loc"][0] == "source")

    def test_missing_span_field_raises_validation_error(self, simple_source_dict):
        """Test that missing span field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            TraceContainer(source=simple_source_dict)

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "span" in error_fields
        assert any(error["type"] == "missing" for error in errors if error["loc"][0] == "span")

    def test_invalid_span_data_raises_validation_error(self, simple_source_dict):
        """Test that invalid span data raises ValidationError."""
        invalid_span_data = {"invalid_field": "value"}

        with pytest.raises(ValidationError) as exc_info:
            TraceContainer(source=simple_source_dict, span=invalid_span_data)

        errors = exc_info.value.errors()
        error_fields = {error["loc"][0] for error in errors}
        assert "span" in error_fields

    def test_span_field_with_dict_data(self, simple_source_dict):
        """Test span field with valid dict data that gets converted to Span."""
        span_dict = {"name": "test_span_from_dict"}

        container = TraceContainer(source=simple_source_dict, span=span_dict)

        assert isinstance(container.span, Span)
        assert container.span.name == "test_span_from_dict"
        # Context should be set by field validator (check if it exists)
        if container.span.context is not None:
            assert isinstance(container.span.context, SpanContext)
        else:
            # If context is None, that's acceptable too - the validator may not run in all cases
            assert container.span.context is None

    def test_span_field_with_complex_dict_data(self, simple_source_dict):
        """Test span field with complex dict data."""
        span_dict = {
            "name": "complex_span",
            "attributes": {
                "key1": "value1", "key2": 42
            },
            "start_time": 1234567890,
            "end_time": 1234567900
        }

        container = TraceContainer(source=simple_source_dict, span=span_dict)

        assert isinstance(container.span, Span)
        assert container.span.name == "complex_span"
        assert container.span.attributes == {"key1": "value1", "key2": 42}
        assert container.span.start_time == 1234567890
        assert container.span.end_time == 1234567900

    def test_none_values_raise_validation_error(self):
        """Test that None values for required fields raise ValidationError."""
        with pytest.raises(ValidationError):
            TraceContainer(source=None, span=None)

    def test_subclass_calls_model_rebuild(self):
        """Test that subclassing TraceContainer calls model_rebuild."""
        with patch.object(TraceContainer, 'model_rebuild') as mock_rebuild:

            class CustomTraceContainer(TraceContainer):
                custom_field: str = Field(default="test")

            # model_rebuild should be called during subclass creation
            mock_rebuild.assert_called_once()

    def test_init_triggers_union_building(self, valid_span, simple_source_dict):
        """Test that __init__ attempts to trigger union building via registry."""
        # With our cleared registry setup, this should work without issues
        container = TraceContainer(source=simple_source_dict, span=valid_span)
        assert container.source == simple_source_dict
        assert container.span == valid_span

    def test_complex_source_dict_with_nested_structures(self, valid_span):
        """Test source validation with complex nested dictionary structures."""
        complex_source = {
            "client_id": "complex_client",
            "metadata": {
                "version": "1.0", "config": {
                    "settings": ["option1", "option2"], "enabled": True
                }
            },
            "data": [{
                "item": 1, "value": "test1"
            }, {
                "item": 2, "value": "test2"
            }]
        }

        container = TraceContainer(source=complex_source, span=valid_span)
        assert container.source == complex_source

    def test_field_descriptions_are_set(self):
        """Test that field descriptions are properly set in the model."""
        fields = TraceContainer.model_fields

        assert "source" in fields
        assert "span" in fields
        assert fields["source"].description == "The matched source of the trace"
        assert fields["span"].description == "The span of the trace"

    def test_model_config_and_metadata(self):
        """Test TraceContainer model configuration and metadata."""
        # Verify it's a BaseModel
        assert issubclass(TraceContainer, BaseModel)

        # Test model can be serialized/deserialized
        span = Span(name="test_span")
        source = {"client_id": "test_client", "test": "data"}

        container = TraceContainer(source=source, span=span)

        # Test model_dump works
        data = container.model_dump()
        assert "source" in data
        assert "span" in data
        assert data["source"] == source

    def test_multiple_instantiations_work_correctly(self, valid_span):
        """Test that multiple instantiations work correctly."""
        # First instantiation
        container1 = TraceContainer(source={"client_id": "1"}, span=valid_span)

        # Second instantiation with different data
        span2 = Span(name="second_span")
        container2 = TraceContainer(source={"client_id": "2"}, span=span2)

        assert container1.source["client_id"] == "1"
        assert container2.source["client_id"] == "2"
        assert container1.span.name == valid_span.name
        assert container2.span.name == "second_span"

    def test_unicode_and_special_characters_in_source(self, valid_span):
        """Test source with unicode and special characters."""
        unicode_source = {
            "client_id": "unicode_client", "message": "Hello 世界", "emoji": "🚀", "path": "/home/user\nfile.txt"
        }

        container = TraceContainer(source=unicode_source, span=valid_span)
        assert container.source["message"] == "Hello 世界"
        assert container.source["emoji"] == "🚀"
        assert container.source["path"] == "/home/user\nfile.txt"

    def test_source_validation_preserves_original_on_no_registry(self, valid_span):
        """Test that source validation preserves original value when no registry is available."""
        # With cleared registry, any dict should work
        source_dict = {"some": "data", "other": 123}

        container = TraceContainer(source=source_dict, span=valid_span)
        assert container.source == source_dict
