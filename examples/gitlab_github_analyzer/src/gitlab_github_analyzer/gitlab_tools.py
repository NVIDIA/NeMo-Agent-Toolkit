# SPDX-FileCopyrightText: Copyright (c) 2025, Your Organization. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
import urllib.parse
from typing import Optional

import httpx
from pydantic import Field

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class GitlabGetFileConfig(FunctionBaseConfig, name="gitlab_getfile"):
    """
    Tool that reads files from GitLab repositories using GitLab URLs.
    Supports private GitLab instances and requires API token for private repos.
    """
    gitlab_instance: str = Field(
        default="https://gitlab.com",
        description="GitLab instance URL (e.g., https://gitlab-master.nvidia.com)"
    )
    api_token: Optional[str] = Field(
        default=None,
        description="GitLab API token for private repositories"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")


@register_function(config_type=GitlabGetFileConfig)
async def gitlab_getfile(config: GitlabGetFileConfig, builder: Builder):
    
    def parse_gitlab_url(url: str) -> dict:
        """Parse GitLab URL to extract components."""
        # GitLab URL format: https://gitlab-instance.com/group/project/-/blob/branch/path/file.ext
        pattern = r'https://([^/]+)/(.+?)/-/blob/([^/]+)/(.+)'
        match = re.match(pattern, url)
        
        if not match:
            raise ValueError(f"Invalid GitLab URL format: {url}")
        
        gitlab_host = match.group(1)
        project_path = match.group(2)
        branch = match.group(3)
        file_path = match.group(4)
        
        return {
            "gitlab_instance": f"https://{gitlab_host}",
            "project_path": project_path,
            "branch": branch,
            "file_path": file_path
        }
    
    def build_gitlab_api_url(gitlab_instance: str, project_path: str, file_path: str, branch: str) -> str:
        """Build GitLab API URL for file access."""
        # Encode project path and file path for URL
        encoded_project = urllib.parse.quote(project_path, safe='')
        encoded_file = urllib.parse.quote(file_path, safe='')
        
        # GitLab API endpoint for raw file content
        api_url = f"{gitlab_instance}/api/v4/projects/{encoded_project}/repository/files/{encoded_file}/raw"
        api_url += f"?ref={branch}"
        
        return api_url

    async def read_gitlab_file(gitlab_url: str) -> str:
        """
        Read a file from GitLab repository.
        
        Args:
            gitlab_url (str): GitLab file URL (e.g., https://gitlab.com/group/project/-/blob/main/README.md)
            
        Returns:
            str: File content or error message
        """
        try:
            # Parse the GitLab URL
            url_parts = parse_gitlab_url(gitlab_url)
            
            # Use config instance or URL instance
            gitlab_instance = url_parts["gitlab_instance"]
            if config.gitlab_instance and config.gitlab_instance != "https://gitlab.com":
                gitlab_instance = config.gitlab_instance
            
            # Build API URL
            api_url = build_gitlab_api_url(
                gitlab_instance,
                url_parts["project_path"],
                url_parts["file_path"],
                url_parts["branch"]
            )
            
            # Set up headers
            headers = {"User-Agent": "AIQ-Toolkit-GitLab-Reader/1.0"}
            if config.api_token:
                headers["PRIVATE-TOKEN"] = config.api_token
            
            logger.info(f"Reading GitLab file: {gitlab_url}")
            logger.debug(f"API URL: {api_url}")
            
            # Make API request
            async with httpx.AsyncClient(timeout=config.timeout) as client:
                response = await client.get(api_url, headers=headers)
                
                if response.status_code == 200:
                    content = response.text
                    logger.info(f"Successfully read file: {url_parts['file_path']}")
                    return content
                    
                elif response.status_code == 404:
                    # Try different branch names
                    for alt_branch in ["main", "master", "develop"]:
                        if alt_branch != url_parts["branch"]:
                            alt_api_url = build_gitlab_api_url(
                                gitlab_instance,
                                url_parts["project_path"],
                                url_parts["file_path"],
                                alt_branch
                            )
                            
                            logger.debug(f"Trying alternate branch {alt_branch}: {alt_api_url}")
                            alt_response = await client.get(alt_api_url, headers=headers)
                            
                            if alt_response.status_code == 200:
                                content = alt_response.text
                                logger.info(f"Successfully read file from {alt_branch} branch")
                                return content
                    
                    return f"404: File not found: {url_parts['file_path']}"
                    
                elif response.status_code == 401:
                    return "401: Authentication required. Please provide a valid GitLab API token."
                    
                elif response.status_code == 403:
                    return "403: Access forbidden. Check repository permissions and API token."
                    
                else:
                    return f"Error {response.status_code}: {response.text}"
                    
        except ValueError as e:
            logger.error(f"URL parsing error: {e}")
            return f"Invalid GitLab URL format. Expected format: https://gitlab.com/group/project/-/blob/branch/path/file.ext"
            
        except httpx.TimeoutException:
            logger.error(f"Timeout accessing GitLab API")
            return f"Timeout: Could not access GitLab repository within {config.timeout} seconds"
            
        except Exception as e:
            logger.exception(f"Error reading GitLab file: {e}")
            return f"Error reading GitLab file: {str(e)}"

    yield FunctionInfo.from_fn(
        read_gitlab_file,
        description="Reads files from GitLab repositories using GitLab URLs. Supports both public and private repositories with API token authentication."
    ) 