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
