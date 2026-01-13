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
#

from fastapi import Response

from nat.front_ends.fastapi.api_versioning import VersioningOptions
from nat.front_ends.fastapi.api_versioning import apply_version_headers


def test_apply_version_headers_sets_version_header():
    response = Response()
    opts = VersioningOptions(api_version_header=True, version="3")

    apply_version_headers(response, opts)

    assert response.headers["X-API-Version"] == "3"
    assert "Deprecation" not in response.headers
