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

import json

from nat.front_ends.fastapi.html_snippets.auth_code_grant_cancelled import AUTH_REDIRECT_CANCELLED_POPUP_HTML
from nat.front_ends.fastapi.html_snippets.auth_code_grant_cancelled import build_auth_redirect_cancelled_html


def _safe_json(value: str | None) -> str:
    """Replicate the HTML-safe JSON encoding used by the snippet builders."""
    return json.dumps(value).replace('<', '\\u003c').replace('>', '\\u003e').replace('/', '\\u002f')


def test_auth_redirect_cancelled_popup_html_notifies_and_closes():
    """AUTH_REDIRECT_CANCELLED_POPUP_HTML (popup variant) posts AUTH_CANCELLED and closes the window."""
    assert "AUTH_CANCELLED" in AUTH_REDIRECT_CANCELLED_POPUP_HTML
    assert "window.opener?.postMessage" in AUTH_REDIRECT_CANCELLED_POPUP_HTML
    assert "window.close()" in AUTH_REDIRECT_CANCELLED_POPUP_HTML


def test_build_auth_redirect_cancelled_html_with_return_url():
    """build_auth_redirect_cancelled_html embeds a safe JSON-encoded return URL in the script."""
    return_url = "http://localhost:3000"
    result = build_auth_redirect_cancelled_html(return_url)
    assert _safe_json(return_url) in result
    assert "window.location.replace" in result
    assert "window.history.back()" in result


def test_build_auth_redirect_cancelled_html_without_return_url():
    """build_auth_redirect_cancelled_html falls back to window.history.back() when return_url is None."""
    result = build_auth_redirect_cancelled_html(None)
    assert _safe_json(None) in result
    assert "window.history.back()" in result


def test_build_auth_redirect_cancelled_html_no_oauth_auth_completed_param():
    """build_auth_redirect_cancelled_html must NOT add oauth_auth_completed so the UI handles cancellation."""
    result = build_auth_redirect_cancelled_html("http://localhost:3000")
    assert "oauth_auth_completed" not in result


def test_build_auth_redirect_cancelled_html_url_characters_escaped():
    """build_auth_redirect_cancelled_html HTML-escapes <, >, and / in the JSON value."""
    return_url = "http://example.com/path?foo=bar&baz=<value>"
    result = build_auth_redirect_cancelled_html(return_url)
    assert _safe_json(return_url) in result
    assert "\\u003c" in result
    assert "\\u003e" in result
    assert "\\u002f" in result
    assert "<value>" not in result


def test_build_auth_redirect_cancelled_html_script_tag_cannot_break_out():
    """A </script> sequence in the URL cannot terminate the enclosing script block."""
    return_url = "http://evil.com/</script><script>alert(1)</script>"
    result = build_auth_redirect_cancelled_html(return_url)
    # The injected </script> is escaped; only the template's own closing tag remains
    assert result.count("</script>") == 1
    assert _safe_json(return_url) in result
