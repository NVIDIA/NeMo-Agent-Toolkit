# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Confluence Page Reader Tool
Fetches and returns the content of a Confluence page given its URL.
"""

import logging
import os
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs
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

class ConfluencePageReaderConfig(FunctionBaseConfig, name="confluence_page_reader"):
    """Configuration for Confluence page reader (Bearer Token only)."""
    base_url: Optional[str] = Field(default=None, description="Confluence base URL")
    api_token: Optional[str] = Field(default=None, description="Confluence API token (Bearer token)")
    space_keys: Optional[List[str]] = Field(default=None, description="List of space keys to search in")
    max_results: int = Field(default=10, description="Maximum number of search results to return")
    timeout: int = Field(default=30, description="Request timeout in seconds")

def _get_auth_headers(config: ConfluencePageReaderConfig) -> Dict[str, str]:
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

@register_function(config_type=ConfluencePageReaderConfig)
async def confluence_page_reader(config: ConfluencePageReaderConfig, builder: Builder):
    """Register the Confluence page reader function."""
    
    async def read_page(url: str) -> Dict[str, Any]:
        try:
            base_url = config.base_url or os.getenv('CONFLUENCE_BASE_URL')
            if not base_url:
                return {"status": "error", "error": "Missing base_url"}
            headers = _get_auth_headers(config)
            
            # Extract page ID from URL
            # Expected URL format: https://domain/pages/viewpage.action?pageId=123456
            # or https://domain/display/SPACE/Page+Title
            page_id = None
            if "pageId=" in url:
                page_id = url.split("pageId=")[1].split("&")[0]
            elif "/display/" in url:
                # For display URLs, we need to search by title
                return {"status": "error", "error": "Display URLs not supported yet. Please use direct page URLs with pageId parameter."}
            else:
                return {"status": "error", "error": "Invalid Confluence URL format. Expected URL with pageId parameter."}
            
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

    yield FunctionInfo.from_fn(
        read_page,
        description="Fetches and returns the content of a Confluence page given its URL. Accepts URLs with pageId parameter. Returns full page content and metadata."
    ) 