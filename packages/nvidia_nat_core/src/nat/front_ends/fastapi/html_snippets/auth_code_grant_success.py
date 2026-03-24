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

AUTH_REDIRECT_SUCCESS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Authentication Complete</title>
    <script>
        (function () {
            window.history.replaceState(null, "", window.location.pathname);

            window.opener?.postMessage({ type: 'AUTH_SUCCESS' }, '*');

            window.close();
        })();
    </script>
</head>
<body>
    <p>Authentication complete. You may now close this window.</p>
</body>
</html>
"""

_AUTH_REDIRECT_SUCCESS_HTML_SAME_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
    <title>Authentication Complete</title>
    <script>
        (function () {
            var returnTo = RETURN_URL_PLACEHOLDER;
            if (returnTo) {
                var url = new URL(returnTo);
                url.searchParams.set('oauth_auth_completed', 'true');
                window.location.replace(url.toString());
            } else {
                window.history.back();
            }
        })();
    </script>
</head>
<body>
    <p>Authentication complete. Redirecting&hellip;</p>
</body>
</html>
"""


def build_auth_redirect_success_html(return_url: str | None = None) -> str:
    """Build the same-page authentication success HTML page.

    Args:
        return_url: The URL to redirect to after successful authentication. When
            provided the page navigates there immediately with an ``oauth_auth_completed``
            query parameter so the UI can distinguish a successful return from the user
            pressing back; otherwise it falls back to ``window.history.back()``.

    Returns:
        An HTML string for the post-authentication redirect page.
    """
    safe_json = json.dumps(return_url).replace('<', '\\u003c').replace('>', '\\u003e').replace('/', '\\u002f')
    return _AUTH_REDIRECT_SUCCESS_HTML_SAME_PAGE_TEMPLATE.replace("RETURN_URL_PLACEHOLDER", safe_json)
