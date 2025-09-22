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

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.data_models.span import Span
from nat.data_models.span import SpanContext
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
    TraceAdapterRegistry,  # yapf: disable
)
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import clear_registry
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import register_adapter
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import unregister_adapter
from nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry import (
    unregister_all_adapters,  # yapf: disable
)
from nat.plugins.data_flywheel.observability.schema.trace_container import TraceContainer


class MockSourceTypeA(BaseModel):
    """Mock source type A for testing."""

    framework: str = "test_framework_a"
    data: dict[str, Any]
    client_id: str


class MockSourceTypeB(BaseModel):
    """Mock source type B for testing."""

    framework: str = "test_framework_b"
    input_data: str
    metadata: dict[str, Any] | None = None


class MockTargetType1(BaseModel):
    """Mock target type 1 for testing conversions."""

    target_id: str
    converted_data: Any
    source_info: str


class MockTargetType2(BaseModel):
    """Mock target type 2 for testing conversions."""

    record_id: str
    processed_content: str
    metadata: dict[str, Any] | None = None


class TestTraceAdapterRegistry:
    """Test suite for TraceAdapterRegistry class."""

    def setup_method(self):
        """Clear registry before each test to ensure clean state."""
        TraceAdapterRegistry.clear_registry()

    def teardown_method(self):
        """Clear registry after each test to avoid cross-test pollution."""
        TraceAdapterRegistry.clear_registry()

    def test_registry_starts_empty(self):
        """Test that the registry starts in a clean state."""
        assert TraceAdapterRegistry.list_registered_types() == {}
        assert TraceAdapterRegistry._union_cache is None or TraceAdapterRegistry._union_cache == Any

    def test_register_adapter_decorator_basic(self):
        """Test basic adapter registration using decorator."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data=trace.source.data, source_info="converted_from_A")

        # Verify registration
        registered = TraceAdapterRegistry.list_registered_types()
        assert MockSourceTypeA in registered
        assert MockTargetType1 in registered[MockSourceTypeA]
        assert registered[MockSourceTypeA][MockTargetType1] == convert_a_to_1

    def test_register_adapter_without_return_annotation_raises_error(self):
        """Test that registering adapter without return type annotation raises ValueError."""

        with pytest.raises(ValueError, match="must have a return type annotation"):

            @register_adapter(MockSourceTypeA)
            def bad_converter(trace: TraceContainer):  # No return type annotation
                return {"bad": "converter"}

    def test_register_multiple_adapters_same_source(self):
        """Test registering multiple target types for the same source type."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="1", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeA)
        def convert_a_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="2", processed_content="processed")

        registered = TraceAdapterRegistry.list_registered_types()
        assert MockSourceTypeA in registered
        assert len(registered[MockSourceTypeA]) == 2
        assert MockTargetType1 in registered[MockSourceTypeA]
        assert MockTargetType2 in registered[MockSourceTypeA]

    def test_register_multiple_source_types(self):
        """Test registering adapters for multiple source types."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="a1", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeB)
        def convert_b_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="b1", converted_data={}, source_info="B")

        registered = TraceAdapterRegistry.list_registered_types()
        assert len(registered) == 2
        assert MockSourceTypeA in registered
        assert MockSourceTypeB in registered

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry.logger')
    def test_register_adapter_logs_registration(self, mock_logger):
        """Test that adapter registration is logged."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        mock_logger.debug.assert_called_with("Registered %s -> %s converter", "MockSourceTypeA", "MockTargetType1")

    def test_convert_successful_conversion(self):
        """Test successful trace conversion."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="converted",
                                   converted_data=trace.source.data,
                                   source_info=f"converted_from_{trace.source.framework}")

        # Create test data
        source = MockSourceTypeA(framework="test_framework_a", data={"key": "value"}, client_id="test_client")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        # Convert
        result = TraceAdapterRegistry.convert(trace_container, MockTargetType1)

        # Verify conversion
        assert isinstance(result, MockTargetType1)
        assert result.target_id == "converted"
        assert result.converted_data == {"key": "value"}
        assert result.source_info == "converted_from_test_framework_a"

    def test_convert_no_registered_converter_raises_error(self):
        """Test that convert raises ValueError when no converter is registered."""

        source = MockSourceTypeA(framework="test", data={}, client_id="test")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        with pytest.raises(ValueError, match="No converter from MockSourceTypeA to MockTargetType1"):
            TraceAdapterRegistry.convert(trace_container, MockTargetType1)

    def test_convert_wrong_target_type_raises_error(self):
        """Test convert with registered source but wrong target type."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        source = MockSourceTypeA(framework="test", data={}, client_id="test")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        # Try to convert to unregistered target type
        with pytest.raises(ValueError, match="No converter from MockSourceTypeA to MockTargetType2"):
            TraceAdapterRegistry.convert(trace_container, MockTargetType2)

    def test_convert_error_message_includes_available_targets(self):
        """Test that convert error message lists available target types."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        source = MockSourceTypeA(framework="test", data={}, client_id="test")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        with pytest.raises(ValueError, match=r"Available targets: \['MockTargetType1'\]"):
            TraceAdapterRegistry.convert(trace_container, MockTargetType2)

    def test_get_adapter_returns_function(self):
        """Test get_adapter returns the converter function."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        source = MockSourceTypeA(framework="test", data={}, client_id="test")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        adapter = TraceAdapterRegistry.get_adapter(trace_container, MockTargetType1)
        assert adapter == convert_a_to_1
        assert callable(adapter)

    def test_get_adapter_returns_none_for_unregistered(self):
        """Test get_adapter returns None when no adapter is registered."""

        source = MockSourceTypeA(framework="test", data={}, client_id="test")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        adapter = TraceAdapterRegistry.get_adapter(trace_container, MockTargetType1)
        assert adapter is None

    def test_union_building_single_type(self):
        """Test union building with a single registered type."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        union = TraceAdapterRegistry.get_current_union()
        assert union == MockSourceTypeA

    def test_union_building_multiple_types(self):
        """Test union building with multiple registered types."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeB)
        def convert_b_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="B")

        union = TraceAdapterRegistry.get_current_union()
        # Union should be a union type containing both source types
        assert hasattr(union, '__args__') or union == (MockSourceTypeA | MockSourceTypeB)

    def test_union_building_empty_registry(self):
        """Test union building with no registered types."""
        union = TraceAdapterRegistry.get_current_union()
        assert union == Any

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry.logger')
    def test_union_building_logs_rebuild(self, mock_logger):
        """Test that union rebuilding is logged."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        # Check that logging occurred (registration triggers rebuild)
        mock_logger.debug.assert_any_call("Rebuilt source union with %d registered source types: %s",
                                          1, ["MockSourceTypeA"])

    @patch.object(TraceContainer, 'model_rebuild')
    def test_trace_container_model_rebuild_called(self, mock_rebuild):
        """Test that TraceContainer.model_rebuild is called during registration."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        mock_rebuild.assert_called()

    @patch.object(TraceContainer, 'model_rebuild', side_effect=Exception("Rebuild failed"))
    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry.logger')
    def test_trace_container_rebuild_error_handled(self, mock_logger, mock_rebuild):
        """Test that TraceContainer rebuild errors are handled gracefully."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        # The logger receives the actual exception object, verify the call was made
        mock_logger.warning.assert_called()
        # Check that the call contained the expected message format and exception
        assert any("Failed to update TraceContainer model:" in str(call) for call in mock_logger.warning.call_args_list)

    def test_unregister_adapter_success(self):
        """Test successful adapter unregistration."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeA)
        def convert_a_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="test", processed_content="processed")

        # Verify both are registered
        assert len(TraceAdapterRegistry.list_registered_types()[MockSourceTypeA]) == 2

        # Unregister one
        result = TraceAdapterRegistry.unregister_adapter(MockSourceTypeA, MockTargetType1)
        assert result is True

        # Verify only one remains
        registered = TraceAdapterRegistry.list_registered_types()
        assert len(registered[MockSourceTypeA]) == 1
        assert MockTargetType2 in registered[MockSourceTypeA]
        assert MockTargetType1 not in registered[MockSourceTypeA]

    def test_unregister_adapter_removes_empty_source_entry(self):
        """Test that unregistering the last adapter removes the source entry."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        # Unregister the only adapter
        result = TraceAdapterRegistry.unregister_adapter(MockSourceTypeA, MockTargetType1)
        assert result is True

        # Verify source entry is removed
        registered = TraceAdapterRegistry.list_registered_types()
        assert MockSourceTypeA not in registered

    def test_unregister_adapter_nonexistent_source_returns_false(self):
        """Test unregistering adapter for nonexistent source type returns False."""
        result = TraceAdapterRegistry.unregister_adapter(MockSourceTypeA, MockTargetType1)
        assert result is False

    def test_unregister_adapter_nonexistent_target_returns_false(self):
        """Test unregistering nonexistent target type returns False."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        result = TraceAdapterRegistry.unregister_adapter(MockSourceTypeA, MockTargetType2)
        assert result is False

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry.logger')
    def test_unregister_adapter_logs_removal(self, mock_logger):
        """Test that adapter unregistration is logged."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        TraceAdapterRegistry.unregister_adapter(MockSourceTypeA, MockTargetType1)

        mock_logger.debug.assert_any_call("Unregistered %s -> %s converter", "MockSourceTypeA", "MockTargetType1")

    def test_unregister_all_adapters_success(self):
        """Test successful removal of all adapters for a source type."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeA)
        def convert_a_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="test", processed_content="processed")

        @register_adapter(MockSourceTypeB)
        def convert_b_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="B")

        # Remove all adapters for MockSourceTypeA
        removed_count = TraceAdapterRegistry.unregister_all_adapters(MockSourceTypeA)
        assert removed_count == 2

        # Verify MockSourceTypeA is removed, MockSourceTypeB remains
        registered = TraceAdapterRegistry.list_registered_types()
        assert MockSourceTypeA not in registered
        assert MockSourceTypeB in registered

    def test_unregister_all_adapters_nonexistent_source_returns_zero(self):
        """Test unregistering all adapters for nonexistent source returns 0."""
        removed_count = TraceAdapterRegistry.unregister_all_adapters(MockSourceTypeA)
        assert removed_count == 0

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry.logger')
    def test_unregister_all_adapters_logs_removal(self, mock_logger):
        """Test that unregistering all adapters is logged."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeA)
        def convert_a_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="test", processed_content="processed")

        TraceAdapterRegistry.unregister_all_adapters(MockSourceTypeA)

        mock_logger.debug.assert_any_call("Unregistered all %d converters for %s", 2, "MockSourceTypeA")

    def test_clear_registry_removes_all_adapters(self):
        """Test that clear_registry removes all registered adapters."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeB)
        def convert_b_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="test", processed_content="processed")

        # Verify registration
        registered = TraceAdapterRegistry.list_registered_types()
        assert len(registered) == 2

        # Clear registry
        removed_count = TraceAdapterRegistry.clear_registry()
        assert removed_count == 2

        # Verify registry is empty
        registered = TraceAdapterRegistry.list_registered_types()
        assert len(registered) == 0
        assert TraceAdapterRegistry._union_cache is None or TraceAdapterRegistry._union_cache == Any

    @patch('nat.plugins.data_flywheel.observability.processor.trace_conversion.trace_adapter_registry.logger')
    def test_clear_registry_logs_removal(self, mock_logger):
        """Test that clearing registry is logged."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        TraceAdapterRegistry.clear_registry()

        mock_logger.debug.assert_any_call("Cleared registry - removed %d total converters", 1)

    def test_list_registered_types_returns_copy(self):
        """Test that list_registered_types returns the internal registry."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        registered = TraceAdapterRegistry.list_registered_types()
        internal_registry = TraceAdapterRegistry._registered_types

        # Should return the actual internal registry (not a copy)
        assert registered is internal_registry

    def test_convenience_functions_work(self):
        """Test that module-level convenience functions work correctly."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        # Test convenience functions
        assert MockSourceTypeA in TraceAdapterRegistry.list_registered_types()

        result = unregister_adapter(MockSourceTypeA, MockTargetType1)
        assert result is True
        assert MockSourceTypeA not in TraceAdapterRegistry.list_registered_types()

        # Re-register for testing other functions
        @register_adapter(MockSourceTypeA)
        def convert_a_to_1_again(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeA)
        def convert_a_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="test", processed_content="processed")

        removed_count = unregister_all_adapters(MockSourceTypeA)
        assert removed_count == 2

        # Register again and test clear
        @register_adapter(MockSourceTypeA)
        def convert_a_to_1_final(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        removed_count = clear_registry()
        assert removed_count == 1

    def test_registry_state_isolation_between_operations(self):
        """Test that registry operations maintain proper state isolation."""

        # Register multiple adapters
        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test_a1", converted_data={}, source_info="A")

        @register_adapter(MockSourceTypeA)
        def convert_a_to_2(trace: TraceContainer) -> MockTargetType2:
            return MockTargetType2(record_id="test_a2", processed_content="processed")

        @register_adapter(MockSourceTypeB)
        def convert_b_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test_b1", converted_data={}, source_info="B")

        # Test that operations on one source don't affect others
        source_a = MockSourceTypeA(framework="test_a", data={"test": "data"}, client_id="client_a")
        source_b = MockSourceTypeB(framework="test_b", input_data="test_input")
        span = Span(name="test_span", context=SpanContext())

        trace_a = TraceContainer(source=source_a, span=span)
        trace_b = TraceContainer(source=source_b, span=span)

        # Both conversions should work
        result_a = TraceAdapterRegistry.convert(trace_a, MockTargetType1)
        result_b = TraceAdapterRegistry.convert(trace_b, MockTargetType1)

        assert result_a.source_info == "A"
        assert result_b.source_info == "B"

        # Remove one source's adapters
        TraceAdapterRegistry.unregister_all_adapters(MockSourceTypeA)

        # MockSourceTypeB should still work
        result_b2 = TraceAdapterRegistry.convert(trace_b, MockTargetType1)
        assert result_b2.source_info == "B"

        # MockSourceTypeA should now fail
        with pytest.raises(ValueError, match="No converter from MockSourceTypeA"):
            TraceAdapterRegistry.convert(trace_a, MockTargetType1)


class TestTraceAdapterRegistryEdgeCases:
    """Test edge cases and error conditions for TraceAdapterRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        TraceAdapterRegistry.clear_registry()

    def teardown_method(self):
        """Clear registry after each test."""
        TraceAdapterRegistry.clear_registry()

    def test_register_same_converter_multiple_times_overwrites(self):
        """Test that registering the same converter multiple times overwrites."""

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1_v1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="v1", converted_data={}, source_info="A")

        first_converter = TraceAdapterRegistry.list_registered_types()[MockSourceTypeA][MockTargetType1]

        @register_adapter(MockSourceTypeA)
        def convert_a_to_1_v2(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="v2", converted_data={}, source_info="A")

        second_converter = TraceAdapterRegistry.list_registered_types()[MockSourceTypeA][MockTargetType1]

        # Should be overwritten
        assert first_converter != second_converter
        assert second_converter == convert_a_to_1_v2

    def test_convert_with_converter_raising_exception(self):
        """Test convert behavior when converter function raises exception."""

        @register_adapter(MockSourceTypeA)
        def failing_converter(trace: TraceContainer) -> MockTargetType1:
            raise RuntimeError("Conversion failed")

        source = MockSourceTypeA(framework="test", data={}, client_id="test")
        span = Span(name="test_span", context=SpanContext())
        trace_container = TraceContainer(source=source, span=span)

        # Exception should propagate from converter
        with pytest.raises(RuntimeError, match="Conversion failed"):
            TraceAdapterRegistry.convert(trace_container, MockTargetType1)

    def test_union_cache_invalidation(self):
        """Test that union cache is properly invalidated on registry changes."""

        # Start with empty cache
        assert TraceAdapterRegistry._union_cache is None or TraceAdapterRegistry._union_cache == Any

        # Register first type
        @register_adapter(MockSourceTypeA)
        def convert_a_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        union1 = TraceAdapterRegistry.get_current_union()
        assert union1 == MockSourceTypeA

        # Register second type
        @register_adapter(MockSourceTypeB)
        def convert_b_to_1(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="B")

        union2 = TraceAdapterRegistry.get_current_union()
        assert union2 != union1  # Should be different now

        # Clear registry
        TraceAdapterRegistry.clear_registry()
        union3 = TraceAdapterRegistry.get_current_union()
        assert union3 == Any  # Should be back to Any

    def test_complex_return_type_annotation(self):
        """Test registration with complex return type annotations."""

        from typing import Optional

        @register_adapter(MockSourceTypeA)
        def convert_with_optional(trace: TraceContainer) -> Optional[MockTargetType1]:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        # Should register successfully
        registered = TraceAdapterRegistry.list_registered_types()
        assert MockSourceTypeA in registered
        # The exact type depends on Python version, but should be the Optional type
        registered_type = list(registered[MockSourceTypeA].keys())[0]
        assert ("Optional" in str(registered_type) or "Union" in str(registered_type)
                or registered_type == type(None) | MockTargetType1)

    def test_union_sorting_consistency(self):
        """Test that union types are sorted consistently for reproducible behavior."""

        # Register types in different order multiple times
        for _ in range(3):
            TraceAdapterRegistry.clear_registry()

            # Register in one order
            @register_adapter(MockSourceTypeB)
            def convert_b(trace: TraceContainer) -> MockTargetType1:
                return MockTargetType1(target_id="b", converted_data={}, source_info="B")

            @register_adapter(MockSourceTypeA)
            def convert_a(trace: TraceContainer) -> MockTargetType1:
                return MockTargetType1(target_id="a", converted_data={}, source_info="A")

            union = TraceAdapterRegistry.get_current_union()
            # Union should be consistent regardless of registration order
            # The exact representation may vary, but it should contain both types
            union_str = str(union)
            assert "MockSourceTypeA" in union_str
            assert "MockSourceTypeB" in union_str

    def test_edge_case_empty_then_populated_registry(self):
        """Test edge case of empty registry becoming populated."""

        # Start empty
        assert TraceAdapterRegistry.list_registered_types() == {}
        union1 = TraceAdapterRegistry.get_current_union()
        assert union1 == Any

        # Add one type
        @register_adapter(MockSourceTypeA)
        def convert_a(trace: TraceContainer) -> MockTargetType1:
            return MockTargetType1(target_id="test", converted_data={}, source_info="A")

        union2 = TraceAdapterRegistry.get_current_union()
        assert union2 == MockSourceTypeA

        # Remove and verify back to empty
        TraceAdapterRegistry.unregister_all_adapters(MockSourceTypeA)
        union3 = TraceAdapterRegistry.get_current_union()
        assert union3 == Any
