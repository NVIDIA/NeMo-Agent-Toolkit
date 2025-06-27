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

@register_function(config_type=ConfluenceClientConfig)
async def confluence_client(config: ConfluenceClientConfig, builder: Builder):
    """Register the Confluence API client function."""

    async def search_confluence(query: str) -> Dict[str, Any]:
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

                    if len(clean_content) > 500:
                        clean_content = clean_content[:500] + "..."

                    page_url = f"{base_url}/pages/viewpage.action?pageId={result.get('id')}"

                    formatted_results.append({
                        "id": result.get("id"),
                        "title": result.get("title"),
                        "content": clean_content,
                        "url": page_url,
                        "space": result.get("space", {}).get("name"),
                        "type": result.get("type"),
                        "created": result.get("created"),
                        "last_modified": result.get("version", {}).get("when")
                    })

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

    async def get_confluence_page(page_id: str) -> Dict[str, Any]:
        try:
            base_url = config.base_url or os.getenv('CONFLUENCE_BASE_URL')
            if not base_url:
                return {"status": "error", "error": "Missing base_url"}

            headers = _get_auth_headers(config)

            if base_url.endswith('/'):
                base_url = base_url[:-1]

            page_url = f"{base_url}/rest/api/content/{page_id}"
            params = {
                "expand": "version,space,body.storage,children.page"
            }

            logger.debug(f"Fetching page: {page_url}")
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.get(page_url, headers=headers, params=params)

                if response.status_code != 200:
                    return {
                        "status": "error",
                        "error": f"Page fetch failed: {response.status_code} - {response.text}"
                    }

                page_data = response.json()
                content = page_data.get("body", {}).get("storage", {}).get("value", "")
                clean_content = re.sub(r'<[^>]+>', ' ', content)
                clean_content = re.sub(r'\s+', ' ', clean_content).strip()

                view_url = f"{base_url}/pages/viewpage.action?pageId={page_data.get('id')}"

                return {
                    "status": "success",
                    "id": page_data.get("id"),
                    "title": page_data.get("title"),
                    "content": clean_content,
                    "url": view_url,
                    "space": page_data.get("space", {}).get("name"),
                    "type": page_data.get("type"),
                    "created": page_data.get("created"),
                    "last_modified": page_data.get("version", {}).get("when"),
                    "version": page_data.get("version", {}).get("number")
                }

        except Exception as e:
            logger.exception(f"Error fetching page: {e}")
            return {"status": "error", "error": str(e)}

    async def search_confluence_by_title(title_query: str) -> Dict[str, Any]:
        try:
            base_url = config.base_url or os.getenv('CONFLUENCE_BASE_URL')
            if not base_url:
                return {"status": "error", "error": "Missing base_url"}

            headers = _get_auth_headers(config)

            if base_url.endswith('/'):
                base_url = base_url[:-1]

            search_url = f"{base_url}/rest/api/content/search"
            params = {
                "cql": f'title ~ "{title_query}"',
                "limit": config.max_results,
                "expand": "version,space,body.storage"
            }

            if config.space_keys:
                space_filter = " OR ".join([f'space = "{space}"' for space in config.space_keys])
                params["cql"] = f'({params["cql"]}) AND ({space_filter})'

            logger.debug(f"Searching Confluence by title: {search_url}")
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

                    if len(clean_content) > 300:
                        clean_content = clean_content[:300] + "..."

                    page_url = f"{base_url}/pages/viewpage.action?pageId={result.get('id')}"

                    formatted_results.append({
                        "id": result.get("id"),
                        "title": result.get("title"),
                        "content": clean_content,
                        "url": page_url,
                        "space": result.get("space", {}).get("name"),
                        "type": result.get("type"),
                        "created": result.get("created"),
                        "last_modified": result.get("version", {}).get("when")
                    })

                return {
                    "status": "success",
                    "query": title_query,
                    "total_results": len(formatted_results),
                    "results": formatted_results,
                    "summary": f"Found {len(formatted_results)} pages with title matching: '{title_query}'"
                }

        except Exception as e:
            logger.exception(f"Error searching Confluence by title: {e}")
            return {"status": "error", "error": str(e)}

    yield FunctionInfo.from_fn(
        search_confluence,
        description="Searches Confluence for content matching the query. Returns relevant pages with content snippets and links."
    )

    yield FunctionInfo.from_fn(
        get_confluence_page,
        description="Retrieves detailed content of a specific Confluence page by page ID. Returns full page content and metadata."
    )

    yield FunctionInfo.from_fn(
        search_confluence_by_title,
        description="Searches Confluence for pages with titles matching the query. Useful for finding specific documentation."
    )
