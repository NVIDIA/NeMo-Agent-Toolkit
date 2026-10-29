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

import importlib
import sys
from unittest.mock import patch

import pytest

import nat.tool


def test_nvidia_rag_module_imports_without_langchain_core(monkeypatch: pytest.MonkeyPatch):
    # A plain `pip install nvidia-nat` does not install langchain-core. `nat.tool.register` imports this module before
    # the retriever, server, code execution and memory tools, so if importing it needs langchain-core, none of those
    # tools get registered.
    for module_name in [name for name in sys.modules if name.split(".")[0] == "langchain_core"]:
        monkeypatch.setitem(sys.modules, module_name, None)
    monkeypatch.setitem(sys.modules, "langchain_core", None)
    monkeypatch.delitem(sys.modules, "nat.tool.nvidia_rag", raising=False)
    monkeypatch.delattr(nat.tool, "nvidia_rag", raising=False)

    with patch("nat.cli.type_registry.GlobalTypeRegistry") as mock_global_registry:
        module = importlib.import_module("nat.tool.nvidia_rag")

    registry = mock_global_registry.get.return_value
    registry.register_function.assert_called_once()
    assert registry.register_function.call_args.args[0].config_type is module.NVIDIARAGToolConfig


_RESULTS_RESPONSE = {"results": [{"content": "text", "score": 0.5, "document_name": "guide.pdf"}]}
_CHUNKS_RESPONSE = {"chunks": [{"content": "text", "score": 0.5, "filename": "notes.md"}]}


@pytest.mark.parametrize("data, expected_title", [(_RESULTS_RESPONSE, "guide.pdf"), (_CHUNKS_RESPONSE, "notes.md")],
                         ids=["results", "chunks"])
def test_parse_rag_response_returns_langchain_documents(data: dict, expected_title: str):
    documents_module = pytest.importorskip("langchain_core.documents")
    from nat.tool.nvidia_rag import parse_rag_response

    documents = parse_rag_response(data)

    assert len(documents) == 1
    assert isinstance(documents[0], documents_module.Document)
    assert documents[0].page_content == "text"
    assert documents[0].metadata == {
        "document_title": expected_title,
        "document_url": "nemo_framework",
        "document_full_text": "text",
        "score_rerank": 0.5,
    }
