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
from unittest.mock import patch

import pytest

from nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin import ElasticsearchMixin


class MockParentClass:
    """Mock parent class for testing mixin inheritance."""

    def __init__(self, *args, **kwargs):
        self.parent_init_called = True
        self.parent_args = args
        self.parent_kwargs = kwargs


class ConcreteElasticsearchMixin(ElasticsearchMixin, MockParentClass):
    """Concrete implementation of ElasticsearchMixin for testing."""
    pass


class TestElasticsearchMixin:
    """Test cases for ElasticsearchMixin class."""

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_mixin_initialization_default_headers(self, mock_elasticsearch):
        """Test ElasticsearchMixin initialization with default headers."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        # Test initialization with default headers
        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Verify initialization
        assert mixin is not None
        assert mixin._index == 'test_index'
        assert mixin._elastic_client == mock_client
        assert mixin.parent_init_called is True

        # Verify AsyncElasticsearch was called with correct parameters
        mock_elasticsearch.assert_called_once_with(
            'http://localhost:9200',
            basic_auth=('user', 'pass'),
            headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_mixin_initialization_custom_headers(self, mock_elasticsearch):
        """Test ElasticsearchMixin initialization with custom headers."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        custom_headers = {"Custom-Header": "custom-value", "X-Test": "true"}

        # Test initialization with custom headers
        mixin = ConcreteElasticsearchMixin(endpoint='https://es.example.com:9200',
                                           index='custom_index',
                                           elasticsearch_auth=('admin', 'secret'),
                                           headers=custom_headers)

        # Verify initialization
        assert mixin is not None
        assert mixin._index == 'custom_index'
        assert mixin._elastic_client == mock_client

        # Verify AsyncElasticsearch was called with custom headers
        mock_elasticsearch.assert_called_once_with('https://es.example.com:9200',
                                                   basic_auth=('admin', 'secret'),
                                                   headers=custom_headers)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_mixin_initialization_with_parent_args(self, mock_elasticsearch):
        """Test ElasticsearchMixin initialization passes args/kwargs to parent class."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        parent_args = ('arg1', 'arg2')
        parent_kwargs = {'parent_param1': 'value1', 'parent_param2': 'value2'}

        # Test initialization with parent class parameters
        mixin = ConcreteElasticsearchMixin(
            *parent_args,
            endpoint='http://localhost:9200',
            index='test_index',
            elasticsearch_auth=('user', 'pass'),
            **parent_kwargs)  # type: ignore  # parent_kwargs expansion confuses type checker

        # Verify parent initialization
        assert mixin.parent_init_called is True
        assert mixin.parent_args == parent_args
        assert mixin.parent_kwargs == parent_kwargs

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_single_document(self, mock_elasticsearch):
        """Test export_processed with a single document."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test single document export
        test_doc = {"field1": "value1", "field2": "value2", "timestamp": 123456789}
        await mixin.export_processed(test_doc)

        # Verify elasticsearch client's index method was called
        mock_client.index.assert_called_once_with(index='test_index', document=test_doc)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_bulk_documents(self, mock_elasticsearch):
        """Test export_processed with bulk documents (list)."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='bulk_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test bulk document export
        test_docs = [{
            "field": "value1", "timestamp": 123456789
        }, {
            "field": "value2", "timestamp": 123456790
        }, {
            "field": "value3", "timestamp": 123456791
        }]
        await mixin.export_processed(test_docs)

        # Verify elasticsearch client's bulk method was called with correct format
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
        }, {
            "index": {
                "_index": "bulk_index"
            }
        }, {
            "field": "value3", "timestamp": 123456791
        }]
        mock_client.bulk.assert_called_once_with(operations=expected_operations)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_empty_list(self, mock_elasticsearch):
        """Test export_processed with empty list (should return without calling elasticsearch)."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test empty list
        await mixin.export_processed([])

        # Verify no elasticsearch calls were made
        mock_client.index.assert_not_called()
        mock_client.bulk.assert_not_called()

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_invalid_item_type(self, mock_elasticsearch):
        """Test export_processed with invalid item types."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test invalid string type
        with pytest.raises(ValueError, match="Invalid item type"):
            await mixin.export_processed("invalid_string")  # type: ignore  # Intentional type error

        # Test invalid integer type
        with pytest.raises(ValueError, match="Invalid item type"):
            await mixin.export_processed(12345)  # type: ignore  # Intentional type error

        # Test invalid None type
        with pytest.raises(ValueError, match="Invalid item type"):
            await mixin.export_processed(None)  # type: ignore  # Intentional type error

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_mixed_list_types(self, mock_elasticsearch):
        """Test export_processed with list containing non-dict items."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test list with mixed types
        with pytest.raises(ValueError, match="All items in list must be dictionaries"):
            await mixin.export_processed([{
                "valid": "dict"
            }, "invalid_string", {
                "another": "valid_dict"
            }, 123])  # type: ignore  # Intentional type error

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_list_with_non_dict(self, mock_elasticsearch):
        """Test export_processed with list containing only non-dict items."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test list with only non-dict items
        with pytest.raises(ValueError, match="All items in list must be dictionaries"):
            await mixin.export_processed(["string1", "string2", 123])  # type: ignore  # Intentional type error

    def test_elasticsearch_mixin_missing_required_parameters(self):
        """Test ElasticsearchMixin initialization fails without required parameters."""
        # Test missing endpoint - should raise TypeError for missing required keyword argument
        with pytest.raises(TypeError):
            ConcreteElasticsearchMixin(  # type: ignore  # Missing required parameter
                index='test_index', elasticsearch_auth=('user', 'pass'))

        # Test missing index - should raise TypeError for missing required keyword argument
        with pytest.raises(TypeError):
            ConcreteElasticsearchMixin(  # type: ignore  # Missing required parameter
                endpoint='http://localhost:9200',
                elasticsearch_auth=('user', 'pass'))

        # Test missing elasticsearch_auth - should raise TypeError for missing required keyword argument
        with pytest.raises(TypeError):
            ConcreteElasticsearchMixin(  # type: ignore  # Missing required parameter
                endpoint='http://localhost:9200', index='test_index')

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_elasticsearch_client_initialization_failure(self, mock_elasticsearch):
        """Test behavior when AsyncElasticsearch initialization fails."""
        # Setup mock to raise exception
        mock_elasticsearch.side_effect = Exception("Elasticsearch client initialization failed")

        # Test that exception is propagated
        with pytest.raises(Exception, match="Elasticsearch client initialization failed"):
            ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                       index='test_index',
                                       elasticsearch_auth=('user', 'pass'))


class TestElasticsearchMixinErrorHandling:
    """Test error handling and edge cases for ElasticsearchMixin."""

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_elasticsearch_client_index_exception(self, mock_elasticsearch):
        """Test behavior when elasticsearch client.index() raises an exception."""
        # Setup mock with exception
        mock_client = AsyncMock()
        mock_client.index.side_effect = Exception("Elasticsearch index error")
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test that exception is propagated
        with pytest.raises(Exception, match="Elasticsearch index error"):
            await mixin.export_processed({"test": "data"})

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_elasticsearch_client_bulk_exception(self, mock_elasticsearch):
        """Test behavior when elasticsearch client.bulk() raises an exception."""
        # Setup mock with exception
        mock_client = AsyncMock()
        mock_client.bulk.side_effect = Exception("Elasticsearch bulk error")
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='test_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test that exception is propagated
        with pytest.raises(Exception, match="Elasticsearch bulk error"):
            await mixin.export_processed([{"test1": "data1"}, {"test2": "data2"}])

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_single_document_with_complex_data(self, mock_elasticsearch):
        """Test export_processed with complex document data."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='complex_index',
                                           elasticsearch_auth=('user', 'pass'))

        # Test with complex nested document
        complex_doc = {
            "metadata": {
                "timestamp": 123456789, "source": "test_system", "tags": ["tag1", "tag2", "tag3"]
            },
            "data": {
                "nested_field": {
                    "value": 42, "type": "integer"
                }, "array_field": [1, 2, 3, 4, 5]
            },
            "message": "This is a test message with unicode: 测试 🚀"
        }

        await mixin.export_processed(complex_doc)

        # Verify elasticsearch client was called with the complex document
        mock_client.index.assert_called_once_with(index='complex_index', document=complex_doc)

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_export_processed_bulk_operations_formatting(self, mock_elasticsearch):
        """Test that bulk operations are formatted correctly for Elasticsearch API."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='formatting_test',
                                           elasticsearch_auth=('user', 'pass'))

        # Test bulk formatting with various document types
        test_docs = [{
            "simple": "document"
        }, {
            "nested": {
                "data": {
                    "value": 123
                }
            }
        }, {
            "array": [1, 2, 3]
        }, {
            "mixed": {
                "string": "value", "number": 42, "boolean": True
            }
        }]

        await mixin.export_processed(test_docs)

        # Verify the exact bulk operations format
        expected_operations = [
            {
                "index": {
                    "_index": "formatting_test"
                }
            },  # Action for doc 1
            {
                "simple": "document"
            },  # Doc 1
            {
                "index": {
                    "_index": "formatting_test"
                }
            },  # Action for doc 2
            {
                "nested": {
                    "data": {
                        "value": 123
                    }
                }
            },  # Doc 2
            {
                "index": {
                    "_index": "formatting_test"
                }
            },  # Action for doc 3
            {
                "array": [1, 2, 3]
            },  # Doc 3
            {
                "index": {
                    "_index": "formatting_test"
                }
            },  # Action for doc 4
            {
                "mixed": {
                    "string": "value", "number": 42, "boolean": True
                }
            }  # Doc 4
        ]

        mock_client.bulk.assert_called_once_with(operations=expected_operations)


class TestElasticsearchMixinIntegration:
    """Integration tests for ElasticsearchMixin."""

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    def test_multiple_mixin_instances_independence(self, mock_elasticsearch):
        """Test that multiple mixin instances are independent."""
        # Setup mocks
        mock_client1 = AsyncMock()
        mock_client2 = AsyncMock()
        mock_elasticsearch.side_effect = [mock_client1, mock_client2]

        # Create two independent instances
        mixin1 = ConcreteElasticsearchMixin(endpoint='http://es1.example.com:9200',
                                            index='index1',
                                            elasticsearch_auth=('user1', 'pass1'),
                                            headers={'X-Client': 'client1'})

        mixin2 = ConcreteElasticsearchMixin(endpoint='http://es2.example.com:9200',
                                            index='index2',
                                            elasticsearch_auth=('user2', 'pass2'),
                                            headers={'X-Client': 'client2'})

        # Verify independence
        assert mixin1._index != mixin2._index
        assert mixin1._elastic_client != mixin2._elastic_client
        assert mixin1._elastic_client == mock_client1
        assert mixin2._elastic_client == mock_client2

        # Verify each client was initialized with correct parameters
        assert mock_elasticsearch.call_count == 2
        calls = mock_elasticsearch.call_args_list

        # First instance
        args1, kwargs1 = calls[0]
        assert args1[0] == 'http://es1.example.com:9200'
        assert kwargs1['basic_auth'] == ('user1', 'pass1')
        assert kwargs1['headers'] == {'X-Client': 'client1'}

        # Second instance
        args2, kwargs2 = calls[1]
        assert args2[0] == 'http://es2.example.com:9200'
        assert kwargs2['basic_auth'] == ('user2', 'pass2')
        assert kwargs2['headers'] == {'X-Client': 'client2'}

    @patch('nat.plugins.data_flywheel.observability.mixin.elasticsearch_mixin.AsyncElasticsearch')
    async def test_sequential_export_operations(self, mock_elasticsearch):
        """Test multiple sequential export operations on the same instance."""
        # Setup mock
        mock_client = AsyncMock()
        mock_elasticsearch.return_value = mock_client

        mixin = ConcreteElasticsearchMixin(endpoint='http://localhost:9200',
                                           index='sequential_test',
                                           elasticsearch_auth=('user', 'pass'))

        # Perform multiple sequential operations
        await mixin.export_processed({"operation": 1})
        await mixin.export_processed([{"operation": 2}, {"operation": 3}])
        await mixin.export_processed({"operation": 4})
        await mixin.export_processed([])  # Empty list
        await mixin.export_processed([{"operation": 5}])

        # Verify all operations were called correctly
        assert mock_client.index.call_count == 2  # Two single document calls
        assert mock_client.bulk.call_count == 2  # Two bulk calls (empty list skipped)

        # Verify individual calls
        index_calls = mock_client.index.call_args_list
        assert index_calls[0][1] == {'index': 'sequential_test', 'document': {"operation": 1}}
        assert index_calls[1][1] == {'index': 'sequential_test', 'document': {"operation": 4}}

        # Verify bulk calls
        bulk_calls = mock_client.bulk.call_args_list
        assert len(bulk_calls) == 2
        # First bulk call (operations 2 and 3)
        assert bulk_calls[0][1]['operations'] == [{
            "index": {
                "_index": "sequential_test"
            }
        }, {
            "operation": 2
        }, {
            "index": {
                "_index": "sequential_test"
            }
        }, {
            "operation": 3
        }]
        # Second bulk call (operation 5)
        assert bulk_calls[1][1]['operations'] == [{"index": {"_index": "sequential_test"}}, {"operation": 5}]
