# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Web fetch action implementation for retrieving and converting web content."""

from __future__ import annotations

import logging
import re
import typing
from urllib.parse import urlparse

from nat.cli.register_workflow import register_workspace_action
from nat.data_models.workspace import ActionContext
from nat.workspace.types import TypeSchema, WorkspaceAction

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_MS = 30_000
MAX_TIMEOUT_MS = 120_000
DEFAULT_MAX_LENGTH = 100_000
MAX_MAX_LENGTH = 500_000

_BLOCKED_HOSTNAMES = frozenset({"localhost", "127.0.0.1"})
_BLOCKED_PREFIXES = ("192.168.", "10.", "172.16.", "172.17.", "172.18.", "172.19.",
                     "172.20.", "172.21.", "172.22.", "172.23.", "172.24.",
                     "172.25.", "172.26.", "172.27.", "172.28.", "172.29.",
                     "172.30.", "172.31.")


def _is_private_host(hostname: str) -> bool:
    """Return True if the hostname resolves to a private/local address."""
    lowered = hostname.lower()
    if lowered in _BLOCKED_HOSTNAMES:
        return True
    return any(lowered.startswith(prefix) for prefix in _BLOCKED_PREFIXES)


def _extract_title(html_content: str) -> str | None:
    """Extract the <title> tag content from HTML."""
    match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html_content, re.IGNORECASE)
    if match:
        import html

        return html.unescape(match.group(1)).strip() or None
    return None


def _extract_description(html_content: str) -> str | None:
    """Extract the meta description from HTML."""
    match = re.search(
        r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
        html_content,
        re.IGNORECASE,
    )
    if not match:
        match = re.search(
            r'<meta[^>]*content=["\']([^"\']*)["\'][^>]*name=["\']description["\']',
            html_content,
            re.IGNORECASE,
        )
    if match:
        import html

        return html.unescape(match.group(1)).strip() or None
    return None


def _html_to_markdown(html_content: str, *, max_length: int) -> str:
    """Convert HTML to readable markdown-like text.

    This is a lightweight conversion that strips tags and preserves structure.
    For full-fidelity conversion, consider using a library like markdownify.
    """
    text = html_content

    # Remove script and style blocks
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<noscript[\s\S]*?</noscript>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<!--[\s\S]*?-->", "", text)

    # Convert headings
    for level in range(1, 7):
        prefix = "#" * level
        text = re.sub(
            rf"<h{level}[^>]*>([\s\S]*?)</h{level}>",
            rf"\n\n{prefix} \1\n\n",
            text,
            flags=re.IGNORECASE,
        )

    # Convert links
    text = re.sub(
        r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>([\s\S]*?)</a>',
        r"[\2](\1)",
        text,
        flags=re.IGNORECASE,
    )

    # Convert paragraphs and line breaks
    text = re.sub(r"<p[^>]*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<hr\s*/?>", "\n---\n", text, flags=re.IGNORECASE)

    # Convert lists
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "", text, flags=re.IGNORECASE)

    # Convert code blocks
    text = re.sub(r"<pre[^>]*>([\s\S]*?)</pre>", r"\n```\n\1\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<code[^>]*>([\s\S]*?)</code>", r"`\1`", text, flags=re.IGNORECASE)

    # Convert bold and italic
    text = re.sub(r"<(?:b|strong)[^>]*>([\s\S]*?)</(?:b|strong)>", r"**\1**", text, flags=re.IGNORECASE)
    text = re.sub(r"<(?:i|em)[^>]*>([\s\S]*?)</(?:i|em)>", r"*\1*", text, flags=re.IGNORECASE)

    # Strip remaining HTML tags
    text = re.sub(r"<[^>]*>", "", text)

    # Decode entities
    import html

    text = html.unescape(text)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    if len(text) > max_length:
        text = text[:max_length] + "\n\n[Content truncated]"

    return text


@register_workspace_action
class WebFetchAction(WorkspaceAction):
    """Fetch content from a URL and return it as readable text."""

    name = "web_fetch"
    description = (
        "Fetch content from a URL and return it as readable markdown (text only).\n"
        "\n"
        "Use this tool to:\n"
        "- Read documentation pages (e.g., API docs, library guides)\n"
        "- Get content from articles or blog posts\n"
        "- Fetch static web page content\n"
        "\n"
        "Features:\n"
        "- Converts HTML to readable markdown automatically\n"
        "- Extracts page title and meta description (include_metadata: true)\n"
        "- Supports custom timeout (default: 30s, max: 120s)\n"
        "- Truncates long content (max_length, default: 100k chars)\n"
        "\n"
        "Limitations:\n"
        "- Cannot render JavaScript or interact with dynamic content\n"
        "- Cannot fetch from localhost or private IP addresses (security)\n"
        "- Only http/https protocols supported\n"
        "\n"
        "For searching the web, use web_search first."
    )
    parameters = [
        TypeSchema(type="string", description="url: URL to fetch (required). Must be http or https."),
        TypeSchema(type="boolean", description="raw: Return raw HTML instead of markdown. Default false."),
        TypeSchema(type="number", description="max_length: Maximum content length in characters (100-500000, default 100000)."),
        TypeSchema(type="number", description="timeout_ms: Request timeout in milliseconds (1000-120000, default 30000)."),
        TypeSchema(type="boolean", description="include_metadata: Extract page title and description. Default true."),
    ]
    result = TypeSchema(
        type="object",
        description="Result with content, title, description, url, content_type, and length.",
    )

    def __init__(self) -> None:
        pass

    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> dict[str, typing.Any]:
        url_str = args.get("url")
        if not isinstance(url_str, str) or not url_str.strip():
            return {"is_error": True, "error": "A non-empty 'url' string is required."}

        # Validate URL
        try:
            parsed = urlparse(url_str)
        except Exception:
            return {"is_error": True, "error": f"Invalid URL: {url_str}"}

        if parsed.scheme not in ("http", "https"):
            return {
                "is_error": True,
                "error": f"Unsupported protocol: {parsed.scheme}. Only http and https are supported.",
            }

        hostname = (parsed.hostname or "").lower()
        if _is_private_host(hostname):
            return {
                "is_error": True,
                "error": "Cannot fetch from localhost or private IP addresses.",
            }

        raw = bool(args.get("raw", False))
        max_length = int(args.get("max_length", DEFAULT_MAX_LENGTH))
        max_length = max(100, min(max_length, MAX_MAX_LENGTH))
        timeout_ms = int(args.get("timeout_ms", DEFAULT_TIMEOUT_MS))
        timeout_ms = max(1000, min(timeout_ms, MAX_TIMEOUT_MS))
        include_metadata = bool(args.get("include_metadata", True))

        try:
            import httpx
        except ImportError:
            return {
                "is_error": True,
                "error": "httpx is required for web_fetch. Install with: pip install httpx",
            }

        try:
            async with httpx.AsyncClient(
                timeout=timeout_ms / 1000.0,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; NATBot/1.0)"},
            ) as client:
                response = await client.get(url_str)
                response.raise_for_status()

            content_type = response.headers.get("content-type", "")
            final_url = str(response.url)
            raw_content = response.text

            # Extract metadata
            title: str | None = None
            description: str | None = None
            if include_metadata and "text/html" in content_type:
                title = _extract_title(raw_content)
                description = _extract_description(raw_content)

            # Convert HTML to markdown or keep raw
            if raw or "text/html" not in content_type:
                content = raw_content
            else:
                content = _html_to_markdown(raw_content, max_length=max_length)

            # Truncate if needed
            if len(content) > max_length:
                content = content[:max_length] + "\n\n[Content truncated]"

            return {
                "content": content,
                "title": title,
                "description": description,
                "url": final_url,
                "content_type": content_type,
                "length": len(content),
            }

        except Exception as exc:
            error_message = str(exc)

            # Provide helpful error messages
            if "ENOTFOUND" in error_message or "getaddrinfo" in error_message or "Name or service not known" in error_message:
                return {
                    "is_error": True,
                    "error": f"DNS lookup failed for {hostname}. The domain may not exist.",
                }
            if "ECONNREFUSED" in error_message or "Connection refused" in error_message:
                return {
                    "is_error": True,
                    "error": f"Connection refused by {hostname}. The server may be down.",
                }
            if "timeout" in error_message.lower() or "timed out" in error_message.lower():
                return {
                    "is_error": True,
                    "error": f"Request timed out after {timeout_ms}ms.",
                }

            return {
                "is_error": True,
                "error": f"Failed to fetch URL: {error_message}",
            }
