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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.plugins.langchain.tools.wikipedia_search import WikiSearchToolConfig


@pytest.fixture(name="tool_config")
def tool_config_fixture() -> WikiSearchToolConfig:
    return WikiSearchToolConfig(max_results=2)


async def test_sets_wikipedia_user_agent(tool_config: WikiSearchToolConfig):
    # Wikipedia's API rejects requests without a User-Agent header (403, non-JSON body).
    # The `wikipedia` package used internally by WikipediaLoader doesn't set one by
    # default, so the tool must configure it explicitly on setup.
    from nat.plugins.langchain.tools.wikipedia_search import wiki_search

    with patch("wikipedia.set_user_agent") as mock_set_user_agent:
        async with wiki_search(tool_config, None):
            pass

        mock_set_user_agent.assert_called_once()
        user_agent = mock_set_user_agent.call_args[0][0]
        assert user_agent.startswith("NeMoAgentToolkit/")
        assert "https://github.com/NVIDIA/NeMo-Agent-Toolkit" in user_agent


async def test_formats_search_results(tool_config: WikiSearchToolConfig):
    from nat.plugins.langchain.tools.wikipedia_search import wiki_search

    mock_doc = MagicMock()
    mock_doc.metadata = {"source": "https://en.wikipedia.org/wiki/Aardvark", "page": ""}
    mock_doc.page_content = "The aardvark is a mammal."

    with patch("wikipedia.set_user_agent"), \
         patch("langchain_community.document_loaders.WikipediaLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.aload = AsyncMock(return_value=[mock_doc])
        mock_loader_cls.return_value = mock_loader

        async with wiki_search(tool_config, None) as func_info:
            result = await func_info.single_fn("aardvark")

        mock_loader_cls.assert_called_once_with(query="aardvark", load_max_docs=tool_config.max_results)
        assert result == ('<Document source="https://en.wikipedia.org/wiki/Aardvark" page=""/>\n'
                          'The aardvark is a mammal.\n</Document>')
