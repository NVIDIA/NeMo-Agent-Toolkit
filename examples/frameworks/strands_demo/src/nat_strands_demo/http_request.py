# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Literal

from strands import tool


@tool
async def http_request(method: Literal["GET"], url: str, convert_to_markdown: bool = False) -> str:
    """Fetch documentation from a URL using an HTTP GET request.

    Args:
        method: HTTP method; this documentation tool supports GET only.
        url: Documentation URL to fetch.
        convert_to_markdown: Convert the response HTML to Markdown when true.

    Returns:
        The response body, optionally converted to Markdown.
    """
    import httpx
    from markdownify import markdownify

    if method != "GET":
        raise ValueError("The documentation tool only supports GET requests.")

    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        response = await client.get(url)
        response.raise_for_status()

    return markdownify(response.text) if convert_to_markdown else response.text
