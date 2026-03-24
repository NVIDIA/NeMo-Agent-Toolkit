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

_AUTH_REDIRECT_CANCELLED_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
    <title>Authorization Cancelled</title>
    <script>
        (function () {
            var returnTo = RETURN_URL_PLACEHOLDER;
            if (returnTo) {
                window.location.replace(returnTo);
            } else {
                window.history.back();
            }
        })();
    </script>
</head>
<body>
    <p>Authorization cancelled. Redirecting&hellip;</p>
</body>
</html>
"""


def build_auth_redirect_cancelled_html(return_url: str | None = None) -> str:
    """Build the authorization-cancelled HTML page.

    Redirects back to the UI without the ``oauth_auth_completed`` query
    parameter so the UI's cancellation-message branch handles it.

    Args:
        return_url: The UI origin to navigate back to. Falls back to
            ``window.history.back()`` when not provided.

    Returns:
        An HTML string for the post-cancellation redirect page.
    """
    safe_json = json.dumps(return_url).replace('<', '\\u003c').replace('>', '\\u003e').replace('/', '\\u002f')
    return _AUTH_REDIRECT_CANCELLED_HTML_TEMPLATE.replace("RETURN_URL_PLACEHOLDER", safe_json)
