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

"""Browser tool for sandbox - unified web browsing and content extraction."""

import json
import logging
import re
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from pydantic import Field

from nat_sandbox_agent.tools.sandbox.executor import SandboxToolExecutor

logger = logging.getLogger(__name__)

def _escape_css_selector(selector: str) -> str:
    """Escape a CSS selector for safe embedding in Python code.

    Args:
        selector: The CSS selector to escape.

    Returns:
        Escaped selector safe for use in f-strings.
    """
    # Escape backslashes first, then quotes
    escaped = selector.replace("\\", "\\\\").replace('"', '\\"')
    # Remove any potential code injection attempts (newlines, etc.)
    escaped = re.sub(r'[\r\n]', ' ', escaped)
    return escaped

class WebBrowseInput(BaseModel):
    """Input schema for web browsing."""

    url: str = Field(
        description="URL to browse."
    )
    selector: str | None = Field(
        default=None,
        description="Optional CSS selector to extract specific elements.",
    )

async def web_browse(
    executor: SandboxToolExecutor,
    url: str,
    selector: str | None = None,
) -> dict[str, Any]:
    """Browse a webpage and extract content.

    This unified tool combines navigation and content extraction:
    - Navigates to the URL
    - Extracts text content (optionally filtered by CSS selector)

    Args:
        executor: The sandbox executor instance.
        url: URL to browse.
        selector: Optional CSS selector to target specific elements.

    Returns:
        Dict with status, page info, and content.
    """
    logger.info(f"Web browse: {url}")

    # Safely escape URL for embedding in generated code
    safe_url = json.dumps(url)

    # Build the content extraction code
    if selector:
        # Escape the selector to prevent code injection
        safe_selector = _escape_css_selector(selector)
        extract_code = f'''
            elements = await page.query_selector_all("{safe_selector}")
            texts = []
            for el in elements:
                text = await el.text_content()
                if text:
                    texts.append(text.strip())
            content = "\\n".join(texts)
'''
    else:
        extract_code = '''
            content = await page.text_content("body") or ""
'''

    code = f'''
import asyncio
import json

async def main():
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            await page.goto({safe_url}, wait_until="domcontentloaded")

            title = await page.title()
            {extract_code}

            result = {{
                "status": "success",
                "url": page.url,
                "title": title,
                "content": content[:{executor.max_output_chars}]
            }}

            await browser.close()
            print(json.dumps(result))

    except Exception as e:
        print(json.dumps({{"status": "error", "error": str(e)}}))

asyncio.run(main())
'''

    # Write script and execute (instead of run_python)
    script_path = "/workspace/temp/_browser_script.py"
    await executor.sandbox.write_file(script_path, code)
    result = await executor.sandbox.run_command(f"python3 {script_path}", timeout=60)

    if result.success and result.stdout.strip():
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            pass

    return {
        "status": "error",
        "error": result.stderr or "Browser operation failed",
    }

def create_web_browse_tool(executor: SandboxToolExecutor) -> StructuredTool:
    """Create the unified web browse tool.

    Args:
        executor: The sandbox executor instance.

    Returns:
        LangChain StructuredTool instance.
    """
    return StructuredTool.from_function(
        coroutine=lambda url, selector=None: web_browse(executor, url, selector),
        name="web_browse",
        description=(
            "Browse a webpage and extract information. "
            "Returns page title and text content. "
            "Use 'selector' to target specific CSS elements."
        ),
        args_schema=WebBrowseInput,
    )
