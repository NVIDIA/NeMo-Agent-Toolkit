import pytest
import asyncio
import aiohttp
import os
import json
from unittest.mock import Mock, patch
import yaml
from pathlib import Path

from library_rag.library_rag_function import library_rag_function, LibraryRagFunctionConfig
from nat.builder.builder import Builder


class TestIntegration:
    """Integration tests for the library RAG function."""

    async def fetch_health_status(self, rag_endpoint, check_dependencies=True) -> bool:
        url = f"{rag_endpoint}/v1/health"
        params = {"check_dependencies": str(check_dependencies)}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                result = await response.json()
        
        if result.get("message") == "Service is up":
            return True
        else:
            print("Basic health check failed")
            return False
        
    
    @pytest.fixture
    def config_from_yaml(self):
        """Load configuration from YAML file for integration testing."""
        # Load the actual config file
        config_path = Path(__file__).parent.parent / "src" / "library_rag" / "configs" / "config.yml"
        
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Extract the function config
        function_config = yaml_config['functions']['library_rag_tool']
        
        # Remove the _type field as it's not part of our config model
        function_config.pop('_type', None)
        
        return LibraryRagFunctionConfig(**function_config)


    @pytest.mark.asyncio
    async def test_end_to_end_workflow_mock(self, config_from_yaml):
        """Test end-to-end workflow with mocked HTTP responses."""
        mock_builder = Mock(spec=Builder)
        
        # Mock a realistic RAG response
        mock_response = {
            "total_results": 2,
            "results": [
                {
                    "document_id": "cuda_guide_1",
                    "content": "CUDA (Compute Unified Device Architecture) is a parallel computing platform and API model created by NVIDIA."
                },
                {
                    "document_id": "cuda_guide_2", 
                    "content": "CUDA allows software developers to use a CUDA-enabled graphics processing unit for general purpose processing."
                }
            ]
        }
        
        from aioresponses import aioresponses
        
        url = f"{config_from_yaml.base_url}/v1/search"
        
        with aioresponses() as mock_http:
            mock_http.post(url, payload=mock_response)
            
            # Initialize the function
            result_generator = library_rag_function(config_from_yaml, mock_builder)
            
            # Get the response function
            response_function = None
            async for item in result_generator:
                if hasattr(item, 'fn'):
                    response_function = item.fn
                    break
            
            assert response_function is not None
            
            # Test a realistic query
            query = "What is CUDA and how does it work?"
            
            results = []
            async for result in response_function(query):
                results.append(result)
            
            # Verify the response structure
            assert len(results) >= 1
            document_content = results[0]
            
            # Check that both documents are included
            assert "cuda_guide_1" in document_content
            assert "cuda_guide_2" in document_content
            assert "CUDA (Compute Unified Device Architecture)" in document_content
            assert "graphics processing unit" in document_content
            
            # Check proper formatting
            assert document_content.count("<Document/>") == 2
            assert document_content.count("</Document>") == 2
            assert "\n\n---\n\n" in document_content

    