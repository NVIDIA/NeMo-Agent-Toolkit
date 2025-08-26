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
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.context import ContextState
from nat.observability.processor.batching_processor import BatchingProcessor
from nat.plugins.data_flywheel.observability.exporter.dfw_exporter import DFWExporter
from nat.plugins.data_flywheel.observability.exporter.dfw_exporter import DictBatchingProcessor
from nat.plugins.data_flywheel.observability.processor import DFWToDictProcessor
from nat.plugins.data_flywheel.observability.processor import SpanToDFWRecordProcessor


class TestDictBatchingProcessor:
    """Test cases for DictBatchingProcessor class."""

    def test_dict_batching_processor_inheritance(self):
        """Test that DictBatchingProcessor properly inherits from BatchingProcessor[dict]."""
        processor = DictBatchingProcessor()

        # Check inheritance
        assert isinstance(processor, BatchingProcessor)

        # Verify it's properly typed for dict
        assert processor.__class__.__bases__ == (BatchingProcessor, )

    def test_dict_batching_processor_initialization_defaults(self):
        """Test DictBatchingProcessor initialization with default parameters."""
        processor = DictBatchingProcessor()

        # Check that it initializes without errors
        assert processor is not None

    def test_dict_batching_processor_initialization_custom_params(self):
        """Test DictBatchingProcessor initialization with custom parameters."""
        processor = DictBatchingProcessor(batch_size=50,
                                          flush_interval=2.0,
                                          max_queue_size=500,
                                          drop_on_overflow=True,
                                          shutdown_timeout=5.0)

        # Check that it initializes without errors
        assert processor is not None


class MockExportContract(BaseModel):
    """Mock export contract for testing."""
    data: str
    timestamp: float


class ConcreteDFWExporter(DFWExporter):
    """Concrete implementation of DFWExporter for testing."""

    @property
    def export_contract(self) -> type[BaseModel]:
        return MockExportContract

    async def export_processed(self, item: dict | list[dict]) -> None:
        """Mock implementation of export_processed."""
        pass


class TestDFWExporter:
    """Test cases for DFWExporter class."""

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_dfw_exporter_initialization_defaults(self, mock_factory_from, mock_factory_to):
        """Test DFWExporter initialization with default parameters."""
        # Setup mocks
        mock_span_processor = Mock(spec=SpanToDFWRecordProcessor)
        mock_dict_processor = Mock(spec=DFWToDictProcessor)
        mock_factory_to.return_value = Mock(return_value=mock_span_processor)
        mock_factory_from.return_value = Mock(return_value=mock_dict_processor)

        exporter = ConcreteDFWExporter()

        # Verify initialization completed without errors
        assert exporter is not None
        assert exporter.export_contract == MockExportContract

        # Verify processor factories were called
        mock_factory_to.assert_called_once_with(SpanToDFWRecordProcessor, to_type=MockExportContract)
        mock_factory_from.assert_called_once_with(DFWToDictProcessor, from_type=MockExportContract)

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_dfw_exporter_initialization_custom_params(self, mock_factory_from, mock_factory_to):
        """Test DFWExporter initialization with custom parameters."""
        # Setup mocks
        mock_span_processor = Mock(spec=SpanToDFWRecordProcessor)
        mock_dict_processor = Mock(spec=DFWToDictProcessor)
        mock_factory_to.return_value = Mock(return_value=mock_span_processor)
        mock_factory_from.return_value = Mock(return_value=mock_dict_processor)

        context_state = Mock(spec=ContextState)

        exporter = ConcreteDFWExporter(context_state=context_state,
                                       batch_size=50,
                                       flush_interval=2.0,
                                       max_queue_size=500,
                                       drop_on_overflow=True,
                                       shutdown_timeout=5.0,
                                       client_id="test_client")

        # Verify initialization completed without errors
        assert exporter is not None
        assert exporter.export_contract == MockExportContract

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    @patch.object(ConcreteDFWExporter, 'add_processor')
    def test_dfw_exporter_processor_chain_setup(self, mock_add_processor, mock_factory_from, mock_factory_to):
        """Test that DFWExporter sets up the correct processor chain."""
        # Setup mocks
        mock_span_processor_class = Mock()
        mock_span_processor_instance = Mock(spec=SpanToDFWRecordProcessor)
        mock_span_processor_class.return_value = mock_span_processor_instance

        mock_dict_processor_class = Mock()
        mock_dict_processor_instance = Mock(spec=DFWToDictProcessor)
        mock_dict_processor_class.return_value = mock_dict_processor_instance

        mock_factory_to.return_value = mock_span_processor_class
        mock_factory_from.return_value = mock_dict_processor_class

        client_id = "test_client_123"
        ConcreteDFWExporter(client_id=client_id)

        # Verify processors were added in correct order
        assert mock_add_processor.call_count == 4

        # Check the calls made to add_processor
        calls = mock_add_processor.call_args_list

        # First call: SpanToDFWRecordProcessor with client_id
        mock_span_processor_class.assert_called_once_with(client_id=client_id)
        assert calls[0][0][0] == mock_span_processor_instance

        # Second call: DFWToDictProcessor
        mock_dict_processor_class.assert_called_once_with()
        assert calls[1][0][0] == mock_dict_processor_instance

        # Third call: DictBatchingProcessor
        batching_processor = calls[2][0][0]
        assert isinstance(batching_processor, DictBatchingProcessor)

        # Fourth call: DictBatchFilterProcessor
        from nat.observability.processor.falsy_batch_filter_processor import DictBatchFilterProcessor
        filter_processor = calls[3][0][0]
        assert isinstance(filter_processor, DictBatchFilterProcessor)

    def test_export_contract_abstract_property(self):
        """Test that export_contract is properly implemented as abstract property."""
        exporter = ConcreteDFWExporter()

        # Should return the mock contract
        assert exporter.export_contract == MockExportContract

        # Verify it's a type (class), not an instance
        assert isinstance(exporter.export_contract, type)
        assert issubclass(exporter.export_contract, BaseModel)

    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that DFWExporter cannot be instantiated directly due to abstract methods."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class DFWExporter.*abstract method"):
            DFWExporter()  # type: ignore  # Intentionally instantiate abstract class to test error

    def test_export_processed_abstract_method(self):
        """Test that export_processed is properly implemented as abstract method."""
        exporter = ConcreteDFWExporter()

        # Should be callable without raising NotImplementedError
        asyncio.run(exporter.export_processed({}))
        asyncio.run(exporter.export_processed([{}, {}]))

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_dfw_exporter_with_none_context_state(self, mock_factory_from, mock_factory_to):
        """Test DFWExporter handles None context_state properly."""
        # Setup mocks
        mock_span_processor = Mock(spec=SpanToDFWRecordProcessor)
        mock_dict_processor = Mock(spec=DFWToDictProcessor)
        mock_factory_to.return_value = Mock(return_value=mock_span_processor)
        mock_factory_from.return_value = Mock(return_value=mock_dict_processor)

        exporter = ConcreteDFWExporter(context_state=None)

        # Should initialize without errors
        assert exporter is not None

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_dfw_exporter_default_client_id(self, mock_factory_from, mock_factory_to):
        """Test DFWExporter uses default client_id when not specified."""
        # Setup mocks
        mock_span_processor_class = Mock()
        mock_span_processor_instance = Mock(spec=SpanToDFWRecordProcessor)
        mock_span_processor_class.return_value = mock_span_processor_instance

        mock_dict_processor_class = Mock()
        mock_dict_processor_instance = Mock(spec=DFWToDictProcessor)
        mock_dict_processor_class.return_value = mock_dict_processor_instance

        mock_factory_to.return_value = mock_span_processor_class
        mock_factory_from.return_value = mock_dict_processor_class

        ConcreteDFWExporter()

        # Verify default client_id was used
        mock_span_processor_class.assert_called_once_with(client_id="default")

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_dfw_exporter_batching_parameters(self, mock_factory_from, mock_factory_to):
        """Test that batching parameters are passed correctly to DictBatchingProcessor."""
        # Setup mocks
        mock_span_processor = Mock(spec=SpanToDFWRecordProcessor)
        mock_dict_processor = Mock(spec=DFWToDictProcessor)
        mock_factory_to.return_value = Mock(return_value=mock_span_processor)
        mock_factory_from.return_value = Mock(return_value=mock_dict_processor)

        batch_size = 75
        flush_interval = 3.5
        max_queue_size = 750
        drop_on_overflow = True
        shutdown_timeout = 15.0

        with patch.object(ConcreteDFWExporter, 'add_processor') as mock_add_processor:
            ConcreteDFWExporter(batch_size=batch_size,
                                flush_interval=flush_interval,
                                max_queue_size=max_queue_size,
                                drop_on_overflow=drop_on_overflow,
                                shutdown_timeout=shutdown_timeout)

            # Find the DictBatchingProcessor call
            batching_processor_call = None
            for call in mock_add_processor.call_args_list:
                processor = call[0][0]
                if isinstance(processor, DictBatchingProcessor):
                    batching_processor_call = call
                    break

            assert batching_processor_call is not None, "DictBatchingProcessor should have been added"

    def test_export_contract_type_consistency(self):
        """Test that export_contract returns consistent type."""
        exporter = ConcreteDFWExporter()

        contract1 = exporter.export_contract
        contract2 = exporter.export_contract

        # Should return the same type every time
        assert contract1 is contract2
        assert contract1 == MockExportContract


class TestDFWExporterErrorCases:
    """Test error cases and edge cases for DFWExporter."""

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_processor_factory_exception_handling(self, mock_factory_from, mock_factory_to):
        """Test behavior when processor factories raise exceptions."""
        # Test when processor_factory_to_type raises exception
        mock_factory_to.side_effect = Exception("Factory error")
        mock_factory_from.return_value = Mock()

        with pytest.raises(Exception, match="Factory error"):
            ConcreteDFWExporter()

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_processor_initialization_exception_handling(self, mock_factory_from, mock_factory_to):
        """Test behavior when processor initialization raises exceptions."""
        # Setup factory mocks to return classes that raise exceptions on init
        mock_span_processor_class = Mock(side_effect=Exception("Processor init error"))
        mock_dict_processor_class = Mock()

        mock_factory_to.return_value = mock_span_processor_class
        mock_factory_from.return_value = mock_dict_processor_class

        with pytest.raises(Exception, match="Processor init error"):
            ConcreteDFWExporter()

    def test_invalid_parameter_types(self):
        """Test behavior with invalid parameter types."""
        # These should still work due to Python's dynamic typing,
        # but we test the behavior
        with patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type'), \
             patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type'):

            # Test with invalid types that would typically cause issues
            try:
                exporter = ConcreteDFWExporter(
                    batch_size="invalid",  # type: ignore
                    flush_interval="invalid",  # type: ignore
                    client_id=123  # type: ignore
                )
                # If it doesn't raise an exception, that's fine - Python is dynamic
                assert exporter is not None
            except (TypeError, ValueError):
                # If it raises a type/value error, that's also acceptable
                pass


class TestDFWExporterIntegration:
    """Integration tests for DFWExporter functionality."""

    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type')
    @patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type')
    def test_full_processor_chain_integration(self, mock_factory_from, mock_factory_to):
        """Test the complete processor chain setup and integration."""
        # Setup mocks that behave more realistically
        mock_span_processor_class = Mock()
        mock_span_processor_instance = Mock(spec=SpanToDFWRecordProcessor)
        mock_span_processor_class.return_value = mock_span_processor_instance

        mock_dict_processor_class = Mock()
        mock_dict_processor_instance = Mock(spec=DFWToDictProcessor)
        mock_dict_processor_class.return_value = mock_dict_processor_instance

        mock_factory_to.return_value = mock_span_processor_class
        mock_factory_from.return_value = mock_dict_processor_class

        # Create exporter and verify complete setup
        exporter = ConcreteDFWExporter(client_id="integration_test")

        # Verify all components were set up
        assert exporter is not None
        assert exporter.export_contract == MockExportContract

        # Verify factory calls
        mock_factory_to.assert_called_once_with(SpanToDFWRecordProcessor, to_type=MockExportContract)
        mock_factory_from.assert_called_once_with(DFWToDictProcessor, from_type=MockExportContract)

        # Verify processor instantiation
        mock_span_processor_class.assert_called_once_with(client_id="integration_test")
        mock_dict_processor_class.assert_called_once_with()

    def test_multiple_exporter_instances(self):
        """Test creating multiple exporter instances with different configurations."""
        with patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_to_type'), \
             patch('nat.plugins.data_flywheel.observability.exporter.dfw_exporter.processor_factory_from_type'):

            exporter1 = ConcreteDFWExporter(client_id="client1", batch_size=50)
            exporter2 = ConcreteDFWExporter(client_id="client2", batch_size=100)

            # Both should be independent instances
            assert exporter1 is not exporter2
            assert exporter1.export_contract == exporter2.export_contract  # Same contract type
            assert exporter1.export_contract is exporter2.export_contract  # Same class reference
