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

from nat.plugins.langchain.tools.tavily_internet_search import TavilyInternetSearchToolConfig
from nat.test import ToolTestRunner


async def test_tavily_internet_search_stub_points_to_external_package():
    runner = ToolTestRunner()

    with pytest.raises(RuntimeError, match="nemo-agent-toolkit-tavily"):
        await runner.test_tool(config_type=TavilyInternetSearchToolConfig, input_data="weather in sf")
