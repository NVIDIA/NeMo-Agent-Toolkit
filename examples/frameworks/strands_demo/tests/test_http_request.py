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

import httpx
import pytest

from nat_strands_demo.http_request import http_request


@pytest.mark.asyncio
async def test_fetch_documentation_as_markdown(httpserver):
    httpserver.expect_request("/docs",
                              method="GET").respond_with_data('<h1>Strands</h1><p><a href="/guide">Read more</a></p>',
                                                              content_type="text/html")
    result = await http_request(method="GET", url=httpserver.url_for("/docs"), convert_to_markdown=True)
    assert "Strands" in result
    assert "[Read more](/guide)" in result
    assert "<h1>" not in result


@pytest.mark.asyncio
async def test_fetch_documentation_without_conversion(httpserver):
    body = "<h1>Strands</h1>"
    httpserver.expect_request("/docs", method="GET").respond_with_data(body, content_type="text/html")
    assert await http_request(method="GET", url=httpserver.url_for("/docs")) == body


@pytest.mark.asyncio
async def test_follow_documentation_redirects(httpserver):
    httpserver.expect_request("/old", method="GET").respond_with_data(status=302, headers={"Location": "/docs"})
    httpserver.expect_request("/docs", method="GET").respond_with_data("Documentation")
    assert await http_request(method="GET", url=httpserver.url_for("/old")) == "Documentation"


@pytest.mark.asyncio
async def test_http_errors_are_reported(httpserver):
    httpserver.expect_request("/missing", method="GET").respond_with_data(status=404)
    with pytest.raises(httpx.HTTPStatusError):
        await http_request(method="GET", url=httpserver.url_for("/missing"))


@pytest.mark.asyncio
async def test_reject_methods_that_modify_documentation():
    with pytest.raises(ValueError, match="only supports GET"):
        await http_request(method="POST", url="https://example.com/docs")
