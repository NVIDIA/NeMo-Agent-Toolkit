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

from nat.plugins.langchain.retriever import milvus_langchain
from nat.retriever.milvus.register import MilvusRetrieverConfig

pytest.importorskip("langchain_milvus")


@pytest.mark.parametrize("top_k, expected_search_kwargs", [(2, {"k": 2}), (None, {})], ids=["top_k", "default"])
async def test_milvus_langchain_passes_top_k(top_k: int | None, expected_search_kwargs: dict):
    config = MilvusRetrieverConfig(uri="http://localhost:19530",
                                   embedding_model="embedder",
                                   collection_name="docs",
                                   top_k=top_k)
    builder = MagicMock()
    builder.get_embedder = AsyncMock(return_value=MagicMock())

    with patch("langchain_milvus.Milvus") as mock_milvus:
        async with milvus_langchain(config, builder) as retriever:
            assert retriever is mock_milvus.return_value.as_retriever.return_value

    mock_milvus.return_value.as_retriever.assert_called_once_with(search_kwargs=expected_search_kwargs)
    milvus_kwargs = mock_milvus.call_args.kwargs
    assert milvus_kwargs["collection_name"] == "docs"
    assert milvus_kwargs["connection_args"] == {"uri": "http://localhost:19530/"}
