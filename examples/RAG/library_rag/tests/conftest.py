import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from aiohttp import ClientSession
from aioresponses import aioresponses

from library_rag.library_rag_function import LibraryRagFunctionConfig
from nat.builder.builder import Builder


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_builder():
    """Mock Builder instance for testing."""
    return Mock(spec=Builder)


@pytest.fixture
def default_config():
    """Default configuration for testing."""
    return LibraryRagFunctionConfig(
        base_url="http://localhost:8081",
        reranker_top_k=2,
        vdb_top_k=10,
        vdb_endpoint="http://milvus:19530",
        collection_names=["test_collection"],
        enable_query_rewriting=True,
        enable_reranker=True
    )


@pytest.fixture
def minimal_config():
    """Minimal configuration for testing."""
    return LibraryRagFunctionConfig(
        base_url="http://localhost:8081"
    )


@pytest.fixture
def sample_rag_response():
    """Sample RAG API response."""
    return {
        "total_results": 2,
        "results": [
            {
                "document_id": "doc_1",
                "content": "This is the first document content about CUDA programming."
            },
            {
                "document_id": "doc_2", 
                "content": "This is the second document content about GPU acceleration."
            }
        ]
    }


@pytest.fixture
def empty_rag_response():
    """Empty RAG API response."""
    return {
        "total_results": 0,
        "results": []
    }


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for testing."""
    with aioresponses() as m:
        yield m
