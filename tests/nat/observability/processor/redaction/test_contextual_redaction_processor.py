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

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from nat.observability.processor.redaction.contextual_redaction_processor import ContextualRedactionProcessor
from nat.observability.processor.redaction.redaction_processor import RedactionContext
from nat.observability.processor.redaction.redaction_processor import RedactionContextState

logger = logging.getLogger(__name__)


def default_callback(_data: Any) -> bool:
    """Default callback that always returns False."""
    return False


class ConcreteContextualRedactionProcessor(ContextualRedactionProcessor[str, dict]):
    """Concrete implementation for testing ContextualRedactionProcessor."""

    def __init__(self,
                 extracted_data: dict | None = None,
                 data_validation_result: bool = True,
                 enabled: bool = True,
                 force_redact: bool = False,
                 redaction_value: str = "[REDACTED]",
                 callback: Callable[..., Any] | None = None,
                 **kwargs):
        if callback is None:
            callback = default_callback
        super().__init__(enabled=enabled,
                         force_redact=force_redact,
                         redaction_value=redaction_value,
                         callback=callback,
                         **kwargs)
        self.extracted_data = extracted_data
        self.data_validation_result = data_validation_result
        self.extract_data_calls = []
        self.validate_data_calls = []
        self.redact_item_calls = []

    def extract_data_from_context(self) -> dict | None:
        """Test implementation that returns configured data."""
        self.extract_data_calls.append(True)
        return self.extracted_data

    def validate_data(self, data: dict) -> bool:
        """Test implementation that returns configured validation result."""
        self.validate_data_calls.append(data)
        return self.data_validation_result

    async def redact_item(self, item: str) -> str:
        """Test implementation that redacts items."""
        self.redact_item_calls.append(item)
        return self.redaction_value


class ErroringContextualRedactionProcessor(ContextualRedactionProcessor[str, dict]):
    """Implementation that raises errors for testing error handling."""

    def __init__(self,
                 extract_error: bool = False,
                 validate_error: bool = False,
                 enabled: bool = True,
                 force_redact: bool = False,
                 redaction_value: str = "[REDACTED]",
                 callback: Callable[..., Any] | None = None,
                 **kwargs):
        if callback is None:
            callback = default_callback
        super().__init__(enabled=enabled,
                         force_redact=force_redact,
                         redaction_value=redaction_value,
                         callback=callback,
                         **kwargs)
        self.extract_error = extract_error
        self.validate_error = validate_error

    def extract_data_from_context(self) -> dict | None:
        """Raises error if configured to do so."""
        if self.extract_error:
            raise RuntimeError("extract_data_from_context failed")
        return {"test": "data"}

    def validate_data(self, data: dict) -> bool:
        """Raises error if configured to do so."""
        if self.validate_error:
            raise ValueError("validate_data failed")
        return True

    async def redact_item(self, item: str) -> str:
        """Test implementation that redacts items."""
        return self.redaction_value


class TestDefaultCallback:
    """Test the default_callback function."""

    def test_default_callback_returns_false(self):
        """Test that default_callback always returns False."""
        assert default_callback("any_data") is False
        assert default_callback(None) is False
        assert default_callback(123) is False
        assert default_callback({"key": "value"}) is False
        assert default_callback([1, 2, 3]) is False

    def test_default_callback_with_various_types(self):
        """Test default_callback with various data types."""
        test_cases = [
            "string",
            123,
            45.67,
            True,
            False,
            None, [], {}, {
                "complex": {
                    "nested": "data"
                }
            }, [1, "mixed", {
                "list": True
            }]
        ]

        for test_case in test_cases:
            assert default_callback(test_case) is False


class TestContextualRedactionProcessorAbstract:
    """Test abstract behavior of ContextualRedactionProcessor."""

    def test_contextual_redaction_processor_is_abstract(self):
        """Test that ContextualRedactionProcessor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ContextualRedactionProcessor()  # type: ignore

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations cannot be instantiated."""

        # Missing both abstract methods
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class IncompleteProcessor(ContextualRedactionProcessor[str, dict]):
                pass

            IncompleteProcessor()  # type: ignore

        # Missing validate_data method
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class MissingValidateData(ContextualRedactionProcessor[str, dict]):

                def extract_data_from_context(self) -> dict | None:
                    return {}

            MissingValidateData()  # type: ignore

        # Missing extract_data_from_context method
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):

            class MissingExtractData(ContextualRedactionProcessor[str, dict]):

                def validate_data(self, data: dict) -> bool:
                    return True

            MissingExtractData()  # type: ignore

    def test_concrete_implementation_can_be_instantiated(self):
        """Test that concrete implementations can be instantiated."""
        processor = ConcreteContextualRedactionProcessor()
        assert isinstance(processor, ContextualRedactionProcessor)
        assert hasattr(processor, 'extract_data_from_context')
        assert hasattr(processor, 'validate_data')
        assert hasattr(processor, 'should_redact')


class TestContextualRedactionProcessorInit:
    """Test ContextualRedactionProcessor initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        processor = ConcreteContextualRedactionProcessor()

        assert processor.callback is default_callback
        assert processor.enabled is True
        assert processor.force_redact is False
        assert processor.redaction_value == "[REDACTED]"
        assert isinstance(processor._redaction_context, RedactionContext)
        assert isinstance(processor._redaction_context._context_state, RedactionContextState)

    def test_custom_callback_initialization(self):
        """Test initialization with custom callback."""

        def custom_callback(data):
            return data.get("sensitive", False)

        processor = ConcreteContextualRedactionProcessor(callback=custom_callback)
        assert processor.callback is custom_callback

    def test_enabled_parameter(self):
        """Test initialization with enabled parameter."""
        processor_enabled = ConcreteContextualRedactionProcessor(enabled=True)
        assert processor_enabled.enabled is True

        processor_disabled = ConcreteContextualRedactionProcessor(enabled=False)
        assert processor_disabled.enabled is False

    def test_force_redact_parameter(self):
        """Test initialization with force_redact parameter."""
        processor_normal = ConcreteContextualRedactionProcessor(force_redact=False)
        assert processor_normal.force_redact is False

        processor_force = ConcreteContextualRedactionProcessor(force_redact=True)
        assert processor_force.force_redact is True

    def test_redaction_value_parameter(self):
        """Test initialization with custom redaction_value."""
        custom_value = "[HIDDEN]"
        processor = ConcreteContextualRedactionProcessor(redaction_value=custom_value)
        assert processor.redaction_value == custom_value

    def test_all_parameters_custom(self):
        """Test initialization with all custom parameters."""

        def custom_callback(data):
            return True

        processor = ConcreteContextualRedactionProcessor(callback=custom_callback,
                                                         enabled=False,
                                                         force_redact=True,
                                                         redaction_value="[CUSTOM]")

        assert processor.callback is custom_callback
        assert processor.enabled is False
        assert processor.force_redact is True
        assert processor.redaction_value == "[CUSTOM]"

    def test_none_callback_uses_default(self):
        """Test that None callback falls back to default_callback."""
        processor = ConcreteContextualRedactionProcessor(callback=None)
        assert processor.callback is default_callback


class TestContextualRedactionProcessorShouldRedact:
    """Test the should_redact method of ContextualRedactionProcessor."""

    async def test_should_redact_force_redact_true(self):
        """Test should_redact when force_redact is True."""
        processor = ConcreteContextualRedactionProcessor(force_redact=True, extracted_data={"test": "data"})

        result = await processor.should_redact("test_item")
        assert result is True

        # Should not call extract_data or validate_data when force_redact is True
        assert len(processor.extract_data_calls) == 0
        assert len(processor.validate_data_calls) == 0

    async def test_should_redact_disabled(self):
        """Test should_redact when processor is disabled."""
        processor = ConcreteContextualRedactionProcessor(enabled=False, extracted_data={"test": "data"})

        result = await processor.should_redact("test_item")
        assert result is False

        # Should not call extract_data or validate_data when disabled
        assert len(processor.extract_data_calls) == 0
        assert len(processor.validate_data_calls) == 0

    async def test_should_redact_no_data_extracted(self):
        """Test should_redact when extract_data_from_context returns None."""
        processor = ConcreteContextualRedactionProcessor(
            extracted_data=None  # Will return None from extract_data_from_context
        )

        result = await processor.should_redact("test_item")
        assert result is False

        assert len(processor.extract_data_calls) == 1
        assert len(processor.validate_data_calls) == 0  # Should not validate if no data

    async def test_should_redact_invalid_data(self):
        """Test should_redact when validate_data returns False."""
        test_data = {"invalid": "data"}
        processor = ConcreteContextualRedactionProcessor(extracted_data=test_data, data_validation_result=False)

        result = await processor.should_redact("test_item")
        assert result is False

        assert len(processor.extract_data_calls) == 1
        assert len(processor.validate_data_calls) == 1
        assert processor.validate_data_calls[0] == test_data

    async def test_should_redact_valid_data_callback_false(self):
        """Test should_redact with valid data but callback returns False."""
        test_data = {"test": "data"}

        def callback_returns_false(data):
            return False

        processor = ConcreteContextualRedactionProcessor(callback=callback_returns_false,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        result = await processor.should_redact("test_item")
        assert result is False

        assert len(processor.extract_data_calls) == 1
        assert len(processor.validate_data_calls) == 1
        assert processor.validate_data_calls[0] == test_data

    async def test_should_redact_valid_data_callback_true(self):
        """Test should_redact with valid data and callback returns True."""
        test_data = {"sensitive": "information"}

        def callback_returns_true(data):
            return True

        processor = ConcreteContextualRedactionProcessor(callback=callback_returns_true,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        result = await processor.should_redact("test_item")
        assert result is True

        assert len(processor.extract_data_calls) == 1
        assert len(processor.validate_data_calls) == 1
        assert processor.validate_data_calls[0] == test_data

    async def test_should_redact_async_callback(self):
        """Test should_redact with async callback."""
        test_data = {"async": "test"}

        async def async_callback(data):
            await asyncio.sleep(0.001)  # Simulate async work
            return data.get("async") == "test"

        processor = ConcreteContextualRedactionProcessor(callback=async_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        result = await processor.should_redact("test_item")
        assert result is True

    async def test_should_redact_callback_with_data_parameter(self):
        """Test that callback receives the correct data parameter."""
        test_data = {"key": "value", "sensitive": True}
        received_data = None

        def capturing_callback(data):
            nonlocal received_data
            received_data = data
            return data.get("sensitive", False)

        processor = ConcreteContextualRedactionProcessor(callback=capturing_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        result = await processor.should_redact("test_item")
        assert result is True
        assert received_data == test_data


class TestContextualRedactionProcessorCaching:
    """Test context-aware caching functionality."""

    async def test_callback_caching_within_context(self):
        """Test that callback results are cached within the same context."""
        call_count = 0
        test_data = {"test": "data"}

        def counting_callback(data):
            nonlocal call_count
            call_count += 1
            return True

        processor = ConcreteContextualRedactionProcessor(callback=counting_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        # First call should execute callback
        result1 = await processor.should_redact("item1")
        assert result1 is True
        assert call_count == 1

        # Second call should use cached result
        result2 = await processor.should_redact("item2")
        assert result2 is True
        assert call_count == 1  # Should not increment

        # extract_data and validate_data should still be called for each item
        assert len(processor.extract_data_calls) == 2
        assert len(processor.validate_data_calls) == 2

    async def test_cache_isolation_between_processors(self):
        """Test that cache is isolated between different processor instances."""
        call_count_1 = 0
        call_count_2 = 0

        def callback_1(data):
            nonlocal call_count_1
            call_count_1 += 1
            return True

        def callback_2(data):
            nonlocal call_count_2
            call_count_2 += 1
            return False

        processor1 = ConcreteContextualRedactionProcessor(callback=callback_1,
                                                          extracted_data={"test": "data1"},
                                                          data_validation_result=True)

        processor2 = ConcreteContextualRedactionProcessor(callback=callback_2,
                                                          extracted_data={"test": "data2"},
                                                          data_validation_result=True)

        # Each processor should execute its own callback
        result1 = await processor1.should_redact("item")
        result2 = await processor2.should_redact("item")

        assert result1 is True
        assert result2 is False
        assert call_count_1 == 1
        assert call_count_2 == 1

    async def test_cache_behavior_with_context_manager(self):
        """Test caching behavior through the context manager."""
        call_count = 0
        test_data = {"cache": "test"}

        def counting_callback(data):
            nonlocal call_count
            call_count += 1
            return data.get("cache") == "test"

        processor = ConcreteContextualRedactionProcessor(callback=counting_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        # Test direct access to context manager
        async with processor._redaction_context.redaction_manager() as manager:
            # First call through manager
            result1 = await manager.redaction_check(counting_callback, test_data)
            assert result1 is True
            assert call_count == 1

            # Second call should use cache
            result2 = await manager.redaction_check(counting_callback, test_data)
            assert result2 is True
            assert call_count == 1  # Should not increment


class TestContextualRedactionProcessorErrorHandling:
    """Test error handling in ContextualRedactionProcessor."""

    async def test_extract_data_error_propagates(self):
        """Test that errors in extract_data_from_context are propagated."""
        processor = ErroringContextualRedactionProcessor(extract_error=True)

        with pytest.raises(RuntimeError, match="extract_data_from_context failed"):
            await processor.should_redact("test_item")

    async def test_validate_data_error_propagates(self):
        """Test that errors in validate_data are propagated."""
        processor = ErroringContextualRedactionProcessor(validate_error=True)

        with pytest.raises(ValueError, match="validate_data failed"):
            await processor.should_redact("test_item")

    async def test_callback_error_propagates(self):
        """Test that errors in callback are propagated."""

        def error_callback(data):
            raise RuntimeError("Callback failed")

        processor = ConcreteContextualRedactionProcessor(callback=error_callback,
                                                         extracted_data={"test": "data"},
                                                         data_validation_result=True)

        with pytest.raises(RuntimeError, match="Callback failed"):
            await processor.should_redact("test_item")

    async def test_async_callback_error_propagates(self):
        """Test that errors in async callback are propagated."""

        async def async_error_callback(data):
            raise ValueError("Async callback failed")

        processor = ConcreteContextualRedactionProcessor(callback=async_error_callback,
                                                         extracted_data={"test": "data"},
                                                         data_validation_result=True)

        with pytest.raises(ValueError, match="Async callback failed"):
            await processor.should_redact("test_item")


class TestContextualRedactionProcessorEdgeCases:
    """Test edge cases and boundary conditions."""

    async def test_should_redact_with_none_extracted_data(self):
        """Test behavior when extract_data_from_context returns None."""
        processor = ConcreteContextualRedactionProcessor(extracted_data=None)

        result = await processor.should_redact("test_item")
        assert result is False

        # Should call extract_data but not validate_data
        assert len(processor.extract_data_calls) == 1
        assert len(processor.validate_data_calls) == 0

    async def test_should_redact_with_empty_dict_data(self):
        """Test behavior with empty dictionary data."""
        empty_data = {}

        def callback_for_empty(data):
            return bool(data)  # Empty dict is falsy

        processor = ConcreteContextualRedactionProcessor(callback=callback_for_empty,
                                                         extracted_data=empty_data,
                                                         data_validation_result=True)

        result = await processor.should_redact("test_item")
        assert result is False

    async def test_should_redact_multiple_calls_same_item(self):
        """Test multiple calls with the same item."""
        call_count = 0
        test_data = {"consistent": "data"}

        def counting_callback(data):
            nonlocal call_count
            call_count += 1
            return True

        processor = ConcreteContextualRedactionProcessor(callback=counting_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        # Multiple calls with same item
        result1 = await processor.should_redact("same_item")
        result2 = await processor.should_redact("same_item")
        result3 = await processor.should_redact("same_item")

        assert result1 is True
        assert result2 is True
        assert result3 is True

        # Callback should only be called once due to caching
        assert call_count == 1

        # But extract_data and validate_data called for each item
        assert len(processor.extract_data_calls) == 3
        assert len(processor.validate_data_calls) == 3

    async def test_precedence_force_redact_over_disabled(self):
        """Test that force_redact takes precedence over enabled=False."""
        processor = ConcreteContextualRedactionProcessor(enabled=False,
                                                         force_redact=True,
                                                         extracted_data={"test": "data"})

        result = await processor.should_redact("test_item")
        assert result is True

        # Should not call extract_data or validate_data when force_redact is True
        assert len(processor.extract_data_calls) == 0
        assert len(processor.validate_data_calls) == 0

    async def test_callback_with_complex_data_types(self):
        """Test callback with complex data types."""
        complex_data = {
            "nested": {
                "deep": {
                    "value": "sensitive"
                }
            },
            "list": [1, 2, {
                "item": "data"
            }],
            "mixed": ["string", 42, {
                "bool": True
            }]
        }

        def complex_callback(data):
            return data.get("nested", {}).get("deep", {}).get("value") == "sensitive"

        processor = ConcreteContextualRedactionProcessor(callback=complex_callback,
                                                         extracted_data=complex_data,
                                                         data_validation_result=True)

        result = await processor.should_redact("test_item")
        assert result is True

    async def test_validate_data_with_different_data_types(self):
        """Test validate_data method with different data types."""

        class TypeValidatingProcessor(ContextualRedactionProcessor[str, Any]):

            def __init__(self,
                         extracted_data: Any = None,
                         enabled: bool = True,
                         force_redact: bool = False,
                         redaction_value: str = "[REDACTED]",
                         callback: Callable[..., Any] | None = None,
                         **kwargs):
                if callback is None:
                    callback = default_callback
                super().__init__(enabled=enabled,
                                 force_redact=force_redact,
                                 redaction_value=redaction_value,
                                 callback=callback,
                                 **kwargs)
                self.extracted_data = extracted_data
                self.validation_calls = []

            def extract_data_from_context(self) -> Any:
                return self.extracted_data

            def validate_data(self, data: Any) -> bool:
                self.validation_calls.append(data)
                # Validate based on type
                return isinstance(data, dict) and bool(data)

            async def redact_item(self, item: str) -> str:
                return self.redaction_value

        # Test with valid dict
        processor1 = TypeValidatingProcessor(extracted_data={"valid": "dict"})
        result1 = await processor1.should_redact("item")
        assert result1 is False  # default_callback returns False
        assert len(processor1.validation_calls) == 1

        # Test with empty dict (invalid)
        processor2 = TypeValidatingProcessor(extracted_data={})
        result2 = await processor2.should_redact("item")
        assert result2 is False
        assert len(processor2.validation_calls) == 1

        # Test with non-dict (invalid)
        processor3 = TypeValidatingProcessor(extracted_data="not_a_dict")
        result3 = await processor3.should_redact("item")
        assert result3 is False
        assert len(processor3.validation_calls) == 1


# =============================================================================
# Test Integration with Redaction Context
# =============================================================================


class TestContextualRedactionProcessorIntegration:
    """Test integration with RedactionContext and RedactionManager."""

    async def test_redaction_context_integration(self):
        """Test integration with the redaction context system."""
        test_data = {"integration": "test"}

        def integration_callback(data):
            return data.get("integration") == "test"

        processor = ConcreteContextualRedactionProcessor(callback=integration_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        # Verify context is properly initialized
        assert isinstance(processor._redaction_context, RedactionContext)
        assert isinstance(processor._redaction_context._context_state, RedactionContextState)

        # Test through should_redact
        result = await processor.should_redact("test_item")
        assert result is True

        # Verify context state was used
        assert processor._redaction_context.redaction_result is True

    async def test_context_state_persistence(self):
        """Test that context state persists across calls."""
        call_count = 0

        def counting_callback(data):
            nonlocal call_count
            call_count += 1
            return True

        processor = ConcreteContextualRedactionProcessor(callback=counting_callback,
                                                         extracted_data={"persistent": "data"},
                                                         data_validation_result=True)

        # First call sets context
        result1 = await processor.should_redact("item1")
        assert result1 is True
        assert call_count == 1

        # Second call uses cached context result
        result2 = await processor.should_redact("item2")
        assert result2 is True
        assert call_count == 1  # Should not increment

        # Verify context state is cached
        assert processor._redaction_context.redaction_result is True

    async def test_manual_context_management(self):
        """Test manual interaction with the context manager."""
        test_data = {"manual": "context"}

        def manual_callback(data):
            return data.get("manual") == "context"

        processor = ConcreteContextualRedactionProcessor(callback=manual_callback,
                                                         extracted_data=test_data,
                                                         data_validation_result=True)

        # Test direct access to context manager
        async with processor._redaction_context.redaction_manager() as manager:
            # Manually call redaction_check
            result = await manager.redaction_check(manual_callback, test_data)
            assert result is True

            # Verify context state
            assert processor._redaction_context.redaction_result is True

            # Clear and test again
            manager.clear_redaction_result()
            assert processor._redaction_context.redaction_result is None

            # New call should execute callback again
            result2 = await manager.redaction_check(manual_callback, test_data)
            assert result2 is True


class TestContextualRedactionProcessorLogging:
    """Test logging behavior."""

    async def test_no_default_logging_in_should_redact(self, caplog):
        """Test that should_redact doesn't log by default."""
        processor = ConcreteContextualRedactionProcessor(extracted_data={"test": "data"}, data_validation_result=True)

        with caplog.at_level(logging.DEBUG):
            await processor.should_redact("test_item")

        # Filter out any logs that come from other parts of the system
        contextual_logs = [record for record in caplog.records if 'contextual_redaction_processor' in record.name]
        assert len(contextual_logs) == 0

    async def test_custom_logging_in_concrete_methods(self, caplog):
        """Test that concrete implementations can add their own logging."""

        class LoggingContextualProcessor(ContextualRedactionProcessor[str, dict]):

            def __init__(self,
                         enabled: bool = True,
                         force_redact: bool = False,
                         redaction_value: str = "[REDACTED]",
                         callback: Callable[..., Any] | None = None,
                         **kwargs):
                if callback is None:
                    callback = default_callback
                super().__init__(enabled=enabled,
                                 force_redact=force_redact,
                                 redaction_value=redaction_value,
                                 callback=callback,
                                 **kwargs)

            def extract_data_from_context(self) -> dict | None:
                logger.info("Extracting data from context")
                return {"logged": "data"}

            def validate_data(self, data: dict) -> bool:
                logger.info("Validating data: %s", data)
                return True

            async def redact_item(self, item: str) -> str:
                return self.redaction_value

        processor = LoggingContextualProcessor()

        with caplog.at_level(logging.INFO):
            await processor.should_redact("test_item")

        # Should see logs from our concrete implementation
        assert "Extracting data from context" in caplog.text
        assert "Validating data: {'logged': 'data'}" in caplog.text


class TestContextualRedactionProcessorPerformance:
    """Test performance-related aspects."""

    async def test_efficient_short_circuit_force_redact(self):
        """Test that force_redact short-circuits efficiently."""

        class ExpensiveProcessor(ContextualRedactionProcessor[str, dict]):

            def __init__(self,
                         enabled: bool = True,
                         force_redact: bool = False,
                         redaction_value: str = "[REDACTED]",
                         callback: Callable[..., Any] | None = None,
                         **kwargs):
                if callback is None:
                    callback = default_callback
                super().__init__(enabled=enabled,
                                 force_redact=force_redact,
                                 redaction_value=redaction_value,
                                 callback=callback,
                                 **kwargs)
                self.extract_calls = 0
                self.validate_calls = 0

            def extract_data_from_context(self) -> dict | None:
                self.extract_calls += 1
                # Simulate expensive operation
                return {"expensive": "operation"}

            def validate_data(self, data: dict) -> bool:
                self.validate_calls += 1
                # Simulate expensive validation
                return True

            async def redact_item(self, item: str) -> str:
                return self.redaction_value

        processor = ExpensiveProcessor(force_redact=True)

        result = await processor.should_redact("test_item")
        assert result is True

        # Should not call expensive operations when force_redact is True
        assert processor.extract_calls == 0
        assert processor.validate_calls == 0

    async def test_efficient_short_circuit_disabled(self):
        """Test that disabled processor short-circuits efficiently."""

        class ExpensiveProcessor(ContextualRedactionProcessor[str, dict]):

            def __init__(self,
                         enabled: bool = True,
                         force_redact: bool = False,
                         redaction_value: str = "[REDACTED]",
                         callback: Callable[..., Any] | None = None,
                         **kwargs):
                if callback is None:
                    callback = default_callback
                super().__init__(enabled=enabled,
                                 force_redact=force_redact,
                                 redaction_value=redaction_value,
                                 callback=callback,
                                 **kwargs)
                self.extract_calls = 0
                self.validate_calls = 0

            def extract_data_from_context(self) -> dict | None:
                self.extract_calls += 1
                return {"expensive": "operation"}

            def validate_data(self, data: dict) -> bool:
                self.validate_calls += 1
                return True

            async def redact_item(self, item: str) -> str:
                return self.redaction_value

        processor = ExpensiveProcessor(enabled=False)

        result = await processor.should_redact("test_item")
        assert result is False

        # Should not call expensive operations when disabled
        assert processor.extract_calls == 0
        assert processor.validate_calls == 0

    async def test_caching_reduces_callback_calls(self):
        """Test that caching reduces expensive callback calls."""
        expensive_call_count = 0

        def expensive_callback(data):
            nonlocal expensive_call_count
            expensive_call_count += 1
            # Simulate expensive operation
            return True

        processor = ConcreteContextualRedactionProcessor(callback=expensive_callback,
                                                         extracted_data={"cached": "data"},
                                                         data_validation_result=True)

        # Multiple calls should only execute callback once
        for i in range(5):
            result = await processor.should_redact(f"item_{i}")
            assert result is True

        # Expensive callback should only be called once
        assert expensive_call_count == 1

        # But extract_data and validate_data called for each item
        assert len(processor.extract_data_calls) == 5
        assert len(processor.validate_data_calls) == 5
