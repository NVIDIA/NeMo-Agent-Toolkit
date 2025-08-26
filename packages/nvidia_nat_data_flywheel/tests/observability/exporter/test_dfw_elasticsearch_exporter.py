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

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from nat.builder.context import AIQContextState
from nat.plugins.data_flywheel.observability.exporter.dfw_elasticsearch_exporter import DFWElasticsearchExporter
from nat.plugins.data_flywheel.observability.schema.sink.elasticsearch import ContractVersion


class MockContractSchema(BaseModel):
    """Mock contract schema for testing."""
    test_field: str
    version: str


class TestDFWElasticsearchExporter:
    """Test cases for DFWElasticsearchExporter class."""

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_exporter_initialization_defaults(self, mock_elasticsearch):
        """Test DFWElasticsearchExporter initialization with default parameters."""
        # Setup mocks

        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch.return_value = mock_elasticsearch_client

        # Required elasticsearch parameters
        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Verify initialization completed without errors
        assert exporter is not None
        assert exporter.contract_version == ContractVersion.V1_1  # default
        assert exporter._index == 'test_index'
        assert exporter._elastic_client == mock_elasticsearch_client

        # Verify elasticsearch client was initialized correctly
        mock_elasticsearch.assert_called_once_with(
            'http://localhost:9200',
            basic_auth=('user', 'pass'),
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_exporter_initialization_custom_params(self, mock_elasticsearch):
        """Test DFWElasticsearchExporter initialization with custom parameters."""
        # Setup mocks

        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch.return_value = mock_elasticsearch_client

        context_state = Mock(spec=AIQContextState)
        custom_headers = {"Custom-Header": "value"}

        exporter = DFWElasticsearchExporter(context_state=context_state,
                                            client_id="test_client",
                                            contract_version=ContractVersion.V1_0,
                                            batch_size=50,
                                            flush_interval=2.0,
                                            max_queue_size=500,
                                            drop_on_overflow=True,
                                            shutdown_timeout=15.0,
                                            endpoint='https://es.example.com:9200',
                                            index='custom_index',
                                            elasticsearch_auth=('admin', 'secret'),
                                            headers=custom_headers)

        # Verify initialization completed without errors
        assert exporter is not None
        assert exporter.contract_version == ContractVersion.V1_0
        assert exporter._index == 'custom_index'
        assert exporter._elastic_client == mock_elasticsearch_client

        # Verify elasticsearch client was initialized with custom parameters
        mock_elasticsearch.assert_called_once_with('https://es.example.com:9200',
                                                   basic_auth=('admin', 'secret'),
                                                   headers=custom_headers)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_export_contract_property(self, mock_elasticsearch):
        """Test that export_contract property delegates to contract_version.get_contract_class()."""
        # Setup mocks

        mock_elasticsearch.return_value = AsyncMock()

        # Mock the contract version to return our mock schema
        mock_contract_version = Mock()
        mock_contract_version.get_contract_class.return_value = MockContractSchema

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200',
            'index': 'test_index',
            'elasticsearch_auth': ('user', 'pass'),
            'contract_version': mock_contract_version
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Test the export_contract property
        contract = exporter.export_contract
        assert contract == MockContractSchema
        # Verify get_contract_class was called (may be called multiple times during initialization)
        assert mock_contract_version.get_contract_class.call_count >= 1

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_export_contract_with_real_enum_values(self, mock_elasticsearch):
        """Test export_contract with real ElasticsearchContractVersion enum values."""
        # Setup mocks

        mock_elasticsearch.return_value = AsyncMock()

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        # Test with VERSION_1_0
        exporter_v1_0 = DFWElasticsearchExporter(contract_version=ContractVersion.V1_0, **elasticsearch_kwargs)
        contract_v1_0 = exporter_v1_0.export_contract
        assert issubclass(contract_v1_0, BaseModel)

        # Test with VERSION_1_1
        exporter_v1_1 = DFWElasticsearchExporter(contract_version=ContractVersion.V1_1, **elasticsearch_kwargs)
        contract_v1_1 = exporter_v1_1.export_contract
        assert issubclass(contract_v1_1, BaseModel)

        # Both should return valid contract classes
        assert contract_v1_0 is not None
        assert contract_v1_1 is not None

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_delegates_to_parent(self, mock_elasticsearch):
        """Test that export_processed delegates to the parent class (ElasticsearchMixin)."""
        # Setup mocks

        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch.return_value = mock_elasticsearch_client

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Test single document export
        test_doc = {"field": "value", "timestamp": 123456789}
        await exporter.export_processed(test_doc)

        # Verify the elasticsearch client's index method was called
        mock_elasticsearch_client.index.assert_called_once_with(index='test_index', document=test_doc)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_bulk_operations(self, mock_elasticsearch):
        """Test export_processed with bulk operations (list of documents)."""
        # Setup mocks

        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch.return_value = mock_elasticsearch_client

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'bulk_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Test bulk document export
        test_docs = [{"field": "value1", "timestamp": 123456789}, {"field": "value2", "timestamp": 123456790}]
        await exporter.export_processed(test_docs)

        # Verify the elasticsearch client's bulk method was called
        expected_operations = [{
            "index": {
                "_index": "bulk_index"
            }
        }, {
            "field": "value1", "timestamp": 123456789
        }, {
            "index": {
                "_index": "bulk_index"
            }
        }, {
            "field": "value2", "timestamp": 123456790
        }]
        mock_elasticsearch_client.bulk.assert_called_once_with(operations=expected_operations)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_exporter_with_none_context_state(self, mock_elasticsearch):
        """Test DFWElasticsearchExporter handles None context_state properly."""
        # Setup mocks

        mock_elasticsearch.return_value = AsyncMock()

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200',
            'index': 'test_index',
            'elasticsearch_auth': ('user', 'pass'),
            'context_state': None
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Should initialize without errors
        assert exporter is not None

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_exporter_headers_default(self, mock_elasticsearch):
        """Test that default headers are applied when none provided."""
        # Setup mocks

        mock_elasticsearch.return_value = AsyncMock()

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Verify default headers were used
        mock_elasticsearch.assert_called_once_with(
            'http://localhost:9200',
            basic_auth=('user', 'pass'),
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})

    def test_missing_required_elasticsearch_parameters(self):
        """Test that missing required elasticsearch parameters raise appropriate errors."""
        with pytest.raises(TypeError):
            # Missing endpoint
            DFWElasticsearchExporter(index='test_index', elasticsearch_auth=('user', 'pass'))

        with pytest.raises(TypeError):
            # Missing index
            DFWElasticsearchExporter(endpoint='http://localhost:9200', elasticsearch_auth=('user', 'pass'))

        with pytest.raises(TypeError):
            # Missing elasticsearch_auth
            DFWElasticsearchExporter(endpoint='http://localhost:9200', index='test_index')


class TestDFWElasticsearchExporterErrorCases:
    """Test error cases and edge cases for DFWElasticsearchExporter."""

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_invalid_item_type(self, mock_elasticsearch):
        """Test export_processed with invalid item types."""
        # Setup mocks

        mock_elasticsearch.return_value = AsyncMock()

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Test with invalid type
        with pytest.raises(ValueError, match="Invalid item type"):
            await exporter.export_processed("invalid_string")  # type: ignore  # Intentional type error for testing

        with pytest.raises(ValueError, match="Invalid item type"):
            await exporter.export_processed(12345)  # type: ignore  # Intentional type error for testing

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_empty_list(self, mock_elasticsearch):
        """Test export_processed with empty list (should return without error)."""
        # Setup mocks

        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch.return_value = mock_elasticsearch_client

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Empty list should not cause errors
        await exporter.export_processed([])

        # Verify no elasticsearch calls were made
        mock_elasticsearch_client.bulk.assert_not_called()
        mock_elasticsearch_client.index.assert_not_called()

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_mixed_list_types(self, mock_elasticsearch):
        """Test export_processed with list containing non-dict items."""
        # Setup mocks

        mock_elasticsearch.return_value = AsyncMock()

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # List with mixed types should raise error
        with pytest.raises(ValueError, match="All items in list must be dictionaries"):
            await exporter.export_processed([{
                "valid": "dict"
            }, "invalid_string", 123])  # type: ignore  # Intentional type error for testing

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_elasticsearch_client_exceptions(self, mock_elasticsearch):
        """Test behavior when Elasticsearch client operations raise exceptions."""
        # Setup mocks

        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch_client.index.side_effect = Exception("Elasticsearch connection error")
        mock_elasticsearch.return_value = mock_elasticsearch_client

        elasticsearch_kwargs = {
            'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
        }

        exporter = DFWElasticsearchExporter(**elasticsearch_kwargs)

        # Exception from elasticsearch client should propagate
        with pytest.raises(Exception, match="Elasticsearch connection error"):
            await exporter.export_processed({"test": "data"})

    def test_elasticsearch_client_initialization_failure(self):
        """Test behavior when Elasticsearch client initialization fails."""
        # Setup mocks

        with patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch',
                   side_effect=Exception("Client init error")):
            elasticsearch_kwargs = {
                'endpoint': 'http://localhost:9200', 'index': 'test_index', 'elasticsearch_auth': ('user', 'pass')
            }

            with pytest.raises(Exception, match="Client init error"):
                DFWElasticsearchExporter(**elasticsearch_kwargs)


class TestDFWElasticsearchExporterIntegration:
    """Integration tests for DFWElasticsearchExporter functionality."""

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_full_initialization_integration(self, mock_elasticsearch):
        """Test complete initialization with both DFWExporter and ElasticsearchMixin functionality."""
        mock_elasticsearch_client = AsyncMock()
        mock_elasticsearch.return_value = mock_elasticsearch_client

        # Create exporter with comprehensive parameters
        exporter = DFWElasticsearchExporter(client_id="integration_test_client",
                                            contract_version=ContractVersion.V1_0,
                                            batch_size=25,
                                            flush_interval=1.5,
                                            max_queue_size=250,
                                            endpoint='http://integration.test:9200',
                                            index='integration_index',
                                            elasticsearch_auth=('test_user', 'test_pass'),
                                            headers={'X-Test': 'integration'})

        # Verify all components were initialized
        assert exporter is not None
        assert exporter.contract_version == ContractVersion.V1_0
        assert exporter._index == 'integration_index'
        assert exporter._elastic_client == mock_elasticsearch_client

        # Verify elasticsearch client initialization
        mock_elasticsearch.assert_called_once_with('http://integration.test:9200',
                                                   basic_auth=('test_user', 'test_pass'),
                                                   headers={'X-Test': 'integration'})

    def test_multiple_exporter_instances_independence(self):
        """Test that multiple exporter instances are independent."""
        with patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch'
                   ) as mock_elasticsearch:  # noqa: E501

            mock_elasticsearch.return_value = AsyncMock()

            exporter1 = DFWElasticsearchExporter(client_id="client1",
                                                 contract_version=ContractVersion.V1_0,
                                                 endpoint='http://es1.test:9200',
                                                 index='index1',
                                                 elasticsearch_auth=('user1', 'pass1'))

            exporter2 = DFWElasticsearchExporter(client_id="client2",
                                                 contract_version=ContractVersion.V1_1,
                                                 endpoint='http://es2.test:9200',
                                                 index='index2',
                                                 elasticsearch_auth=('user2', 'pass2'))

            # Should be independent instances
            assert exporter1 is not exporter2
            assert exporter1.contract_version != exporter2.contract_version
            assert exporter1._index != exporter2._index

            # But should have same contract base type
            assert isinstance(exporter1.export_contract, type)
            assert isinstance(exporter2.export_contract, type)
            assert issubclass(exporter1.export_contract, BaseModel)
            assert issubclass(exporter2.export_contract, BaseModel)
