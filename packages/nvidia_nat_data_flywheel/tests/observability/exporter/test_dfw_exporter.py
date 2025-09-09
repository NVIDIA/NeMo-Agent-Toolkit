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

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.context import ContextState
from nat.observability.processor.batching_processor import BatchingProcessor
from nat.plugins.data_flywheel.observability.exporter.dfw_exporter import DFWExporter
from nat.plugins.data_flywheel.observability.exporter.dfw_exporter import DictBatchingProcessor


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

    def __init__(self, **kwargs):
        # Provide the export contract to parent class
        super().__init__(export_contract=MockExportContract, **kwargs)

    @property
    def export_contract(self) -> type[BaseModel]:
        return MockExportContract

    async def export_processed(self, item: dict | list[dict]) -> None:
        """Mock implementation of export_processed."""
        pass


class TestDFWExporter:
    """Test cases for DFWExporter class."""

    def test_dfw_exporter_initialization_defaults(self):
        """Test DFWExporter initialization with default parameters."""
        exporter = ConcreteDFWExporter()

        # Verify initialization completed without errors
        assert exporter is not None
        assert exporter.export_contract == MockExportContract

    def test_dfw_exporter_initialization_custom_params(self):
        """Test DFWExporter initialization with custom parameters."""
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

    @patch.object(ConcreteDFWExporter, 'add_processor')
    def test_dfw_exporter_processor_chain_setup(self, mock_add_processor):
        """Test that DFWExporter sets up the correct processor chain."""
        client_id = "test_client_123"
        ConcreteDFWExporter(client_id=client_id)

        # Verify processors were added (4 total: span, dict, batching, filter)
        assert mock_add_processor.call_count == 4

    def test_export_contract_property(self):
        """Test that export_contract property returns correct type."""
        exporter = ConcreteDFWExporter()

        contract = exporter.export_contract
        assert contract == MockExportContract
        assert isinstance(contract, type)
        assert issubclass(contract, BaseModel)

    def test_abstract_base_class_cannot_be_instantiated(self):
        """Test that DFWExporter cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            DFWExporter(export_contract=MockExportContract)  # type: ignore

    async def test_export_processed_abstract_method(self):
        """Test that export_processed is properly implemented as abstract method."""
        exporter = ConcreteDFWExporter()

        # This should work without error since it's implemented in concrete class
        await exporter.export_processed({})

    def test_dfw_exporter_with_none_context_state(self):
        """Test DFWExporter handles None context_state properly."""
        exporter = ConcreteDFWExporter(context_state=None)

        # Verify initialization completed without errors
        assert exporter is not None

    def test_dfw_exporter_default_client_id(self):
        """Test DFWExporter uses default client_id when not specified."""
        ConcreteDFWExporter()

        # Should initialize without errors using default client_id
        # This test just verifies no exception is raised

    def test_dfw_exporter_batching_parameters(self):
        """Test that batching parameters are handled correctly."""
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

            # Verify that processors were added
            assert mock_add_processor.call_count == 4

    def test_export_contract_type_consistency(self):
        """Test that export_contract returns consistent type."""
        exporter = ConcreteDFWExporter()

        contract1 = exporter.export_contract
        contract2 = exporter.export_contract

        # Should return same type instance
        assert contract1 == contract2
        assert contract1 is contract2  # Same class reference


class TestDFWExporterErrorCases:
    """Test error cases and edge cases for DFWExporter."""

    def test_invalid_parameter_types(self):
        """Test behavior with invalid parameter types."""
        # These should still work due to Python's dynamic typing
        exporter = ConcreteDFWExporter(
            batch_size="invalid",  # type: ignore[arg-type]
            flush_interval="invalid",  # type: ignore[arg-type]
            max_queue_size="invalid")  # type: ignore[arg-type]

        # Should still initialize (Python is dynamically typed)
        assert exporter is not None


class TestDFWExporterIntegration:
    """Integration tests for DFWExporter functionality."""

    def test_full_processor_chain_integration(self):
        """Test the complete processor chain setup and integration."""
        # Create exporter and verify complete setup
        exporter = ConcreteDFWExporter(client_id="integration_test")

        # Verify exporter was created successfully
        assert exporter is not None
        assert exporter.export_contract == MockExportContract

    def test_multiple_exporter_instances(self):
        """Test creating multiple exporter instances with different configurations."""
        exporter1 = ConcreteDFWExporter(client_id="client1", batch_size=50)
        exporter2 = ConcreteDFWExporter(client_id="client2", batch_size=100)

        # Verify both instances are independent
        assert exporter1 is not exporter2
        assert exporter1.export_contract == exporter2.export_contract  # Same contract type
        assert exporter1.export_contract is exporter2.export_contract  # Same class reference
