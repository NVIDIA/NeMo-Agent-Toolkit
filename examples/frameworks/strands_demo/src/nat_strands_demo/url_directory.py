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

import logging
from urllib.parse import urlparse

from pydantic import Field
from pydantic import field_validator

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

logger = logging.getLogger(__name__)


class URLDirectoryConfig(FunctionBaseConfig, name="url_directory"):
    """Configuration for URL directory tool that provides vetted URLs."""

    urls: dict[str, str] = Field(
        ...,
        description=("Dictionary mapping URL names to URLs (such as "
                     "{'strands_docs': 'https://...', 'api_guide': 'https://...'})"),
    )
    description: str = Field(
        "Get vetted URLs for specific topics or documentation",
        description="Description for when to use this tool",
    )

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, v):
        """Validate that all URLs are properly formatted."""
        for name, url in v.items():
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                raise ValueError(f"Invalid URL for '{name}': {url}")
        return v


@register_function(config_type=URLDirectoryConfig)
async def url_directory(config: URLDirectoryConfig, _: Builder):
    """
    Create a URL directory tool that provides vetted URLs for specific topics.

    This tool acts as a knowledge base directory, providing approved URLs that
    the agent can then fetch using the Strands http_request tool. This prevents
    URL hallucination and ensures the agent only accesses approved domains.
    """

    async def _get_url_directory(query: str) -> str:
        """
        Get the directory of available URLs and their descriptions.

        Args:
            query: The topic or type of URL being requested

        Returns:
            A formatted directory of available URLs with descriptions
        """
        try:
            # Create a formatted directory of URLs
            directory_lines = [
                "Available URLs in the knowledge base:",
                "=" * 40,
            ]

            for name, url in config.urls.items():
                # Parse URL to get domain for context
                parsed = urlparse(url)
                domain = parsed.netloc

                directory_lines.append(f"• {name}:")
                directory_lines.append(f"  URL: {url}")
                directory_lines.append(f"  Domain: {domain}")
                directory_lines.append("")

            directory_lines.extend([
                "Usage Instructions:",
                "1. Choose the appropriate URL from the list above",
                "2. Use the http_request tool to fetch content from that URL",
                ("3. Example: http_request(method='GET', "
                 "url='<selected_url>', convert_to_markdown=True)"),
                "",
                f"Query context: {query}",
            ])

            return "\n".join(directory_lines)

        except Exception as e:
            logger.exception("Error generating URL directory: %s", e)
            return f"Error accessing URL directory: {str(e)}"

    yield FunctionInfo.from_fn(_get_url_directory, description=config.description)
