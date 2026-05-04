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

import pytest
from pydantic import SecretStr


@pytest.mark.parametrize("constructor_args", [{}, {
    "api_key": ""
}, {
    "api_key": "my_api_key"
}],
                         ids=["default", "empty_api_key", "provided_api_key"])
def test_api_key_is_secret_str(constructor_args: dict):
    from nat.plugins.langchain.tools.tavily_internet_search import TavilyInternetSearchToolConfig
    expected_api_key = constructor_args.get("api_key", "")

    config = TavilyInternetSearchToolConfig(**constructor_args)
    assert isinstance(config.api_key, SecretStr)

    api_key = config.api_key.get_secret_value()
    assert api_key == expected_api_key


def test_default_api_key_is_unique_instance():
    from nat.plugins.langchain.tools.tavily_internet_search import TavilyInternetSearchToolConfig

    config1 = TavilyInternetSearchToolConfig()
    config2 = TavilyInternetSearchToolConfig()

    assert config1.api_key is not config2.api_key
