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

import warnings

import pytest


@pytest.fixture(scope="session", autouse=True)
def ignore_warnings():
    """
    Ignore warning about google-cloud-storage deprecation in tests. Remove once issue #1188 is resolved.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=r"^Support for google-cloud-storage")
        yield
