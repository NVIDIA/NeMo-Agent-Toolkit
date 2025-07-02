# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Confluence API client module.
Provides functionality for searching Confluence using read-only API and returning relevant answers with links.
"""

import logging
import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv

import httpx
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

# Import the standalone read function
try:
    from .confluence_page import read_confluence_page, ConfluencePageReaderConfig
except ImportError:
    from confluence_page import read_confluence_page, ConfluencePageReaderConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from known locations
env_paths = [
    Path(__file__).parent.parent.parent / '.env',
    Path.cwd() / '.env',
    Path.home() / '.env'
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        logger.debug(f"Loaded environment from: {env_path}")
        break

class ConfluenceClientConfig(FunctionBaseConfig, name="confluence_client"):
    """Configuration for Confluence API client (Bearer Token only)."""
    base_url: Optional[str] = Field(default=None, description="Confluence base URL")
    api_token: Optional[str] = Field(default=None, description="Confluence API token (Bearer token)")
    space_keys: Optional[List[str]] = Field(default=None, description="List of space keys to search in")
    max_results: int = Field(default=10, description="Maximum number of search results to return")
    timeout: int = Field(default=30, description="Request timeout in seconds")

def _get_auth_headers(config: ConfluenceClientConfig) -> Dict[str, str]:
    """Generate headers using Bearer Token Auth (recommended)."""
    api_token = config.api_token or os.getenv('CONFLUENCE_API_TOKEN')

    if not api_token:
        raise ValueError("Missing required configuration: api_token must be provided")

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_token}"
    }

    logger.debug("Using Bearer token authentication")
    return headers

async def search_confluence(query: str, config: ConfluenceClientConfig) -> Dict[str, Any]:
    """Standalone version of search_confluence that can be imported for testing."""
    try:
        base_url = config.base_url or os.getenv('CONFLUENCE_BASE_URL')
        if not base_url:
            return {"status": "error", "error": "Missing base_url"}

        headers = _get_auth_headers(config)

        if base_url.endswith('/'):
            base_url = base_url[:-1]

        search_url = f"{base_url}/rest/api/content/search"
        params = {
            "cql": f'text ~ "{query}"',
            "limit": config.max_results,
            "expand": "version,space,body.storage"
        }

        if config.space_keys:
            space_filter = " OR ".join([f'space = "{space}"' for space in config.space_keys])
            params["cql"] = f'({params["cql"]}) AND ({space_filter})'

        logger.debug(f"Searching Confluence: {search_url}")
        async with httpx.AsyncClient(timeout=config.timeout) as client:
            response = await client.get(search_url, headers=headers, params=params)

            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"Search failed: {response.status_code} - {response.text}"
                }

            data = response.json()
            results = data.get("results", [])

            formatted_results = []
            for result in results:
                content = result.get("body", {}).get("storage", {}).get("value", "")
                clean_content = re.sub(r'<[^>]+>', ' ', content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()

                page_url = f"{base_url}/pages/viewpage.action?pageId={result.get('id')}"

                formatted_results.append({
                    "id": result.get("id"),
                    "title": result.get("title"),
                    "content": clean_content,
                    "url": page_url,
                })

            # Always fetch full content for each page
            page_reader_config = ConfluencePageReaderConfig(
                base_url=config.base_url,
                api_token=config.api_token,
                timeout=config.timeout
            )
            
            for result in formatted_results:
                full_page_data = await read_confluence_page(result["url"], page_reader_config)
                if full_page_data.get("status") == "success":
                    result["full_content"] = full_page_data.get("content", "")
                else:
                    result["full_content_error"] = full_page_data.get("error", "Unknown error")

            return {
                "status": "success",
                "query": query,
                "total_results": len(formatted_results),
                "results": formatted_results,
                "summary": f"Found {len(formatted_results)} relevant pages for query: '{query}'"
            }

    except Exception as e:
        logger.exception(f"Error searching Confluence: {e}")
        return {"status": "error", "error": str(e)}

@register_function(config_type=ConfluenceClientConfig)
async def confluence_client(config: ConfluenceClientConfig, builder: Builder):
    """Register the Confluence API client function."""

    async def search_confluence_wrapper(query: str) -> Dict[str, Any]:
        """Wrapper function for the registered tool."""
        return await search_confluence(query, config)

    yield FunctionInfo.from_fn(
        search_confluence_wrapper,
        description="Searches Confluence for content matching the query and retrieves full content from all found pages. Returns both individual page details and complete content from each page."
    )
