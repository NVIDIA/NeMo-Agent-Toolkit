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

import pytest

from aiq.retriever.models import AIQDocument
from aiq.retriever.models import RetrieverOutput
from aiq.retriever.models import retriever_output_to_dict
from aiq.retriever.models import retriever_output_to_str


def test_document_methods():
    data = {"page_content": "Here is the document text", "metadata": {"title": "My Document", "type": "test_document"}}

    doc = AIQDocument(page_content="My AIQ Toolkit Document", metadata={})
    assert isinstance(doc, AIQDocument)
    assert doc.page_content == "My AIQ Toolkit Document"
    assert not doc.metadata

    doc = AIQDocument.from_dict(data)
    assert isinstance(doc, AIQDocument)
    assert doc.page_content == data["page_content"]
    assert isinstance(doc.metadata, dict)
    assert doc.document_id is None

    data.update({"document_id": "1234"})
    doc = AIQDocument.from_dict(data)
    assert isinstance(doc, AIQDocument)
    assert doc.page_content == data["page_content"]
    assert isinstance(doc.metadata, dict)
    assert doc.document_id == "1234"

    assert doc.model_dump() == data


@pytest.fixture(name="mock_results_dict", scope="module")
def mock_output_dict():
    return [
        {
            "page_content": "Content for the first document", "metadata": {
                "title": "Doc  1"
            }, "document_id": "135"
        },
        {
            "page_content": "Content for the second document", "metadata": {
                "title": "Doc 2"
            }, "document_id": "246"
        },
    ]


def test_retriever_output(mock_results_dict):
    import json

    output = RetrieverOutput(results=[AIQDocument.from_dict(d) for d in mock_results_dict])
    assert len(output) == 2

    results_dict = retriever_output_to_dict(output)
    assert isinstance(results_dict, dict)
    assert list(results_dict.keys()) == ["results"]
    assert results_dict["results"] == mock_results_dict

    results_str = retriever_output_to_str(output)
    assert isinstance(results_str, str)
    print(results_str)
    assert json.loads(results_str)["results"] == mock_results_dict


def test_validation():
    from pydantic import ValidationError
    data = {"page_content": "Document content"}
    with pytest.raises(ValidationError):
        _ = AIQDocument.from_dict(data)
        data.update({"metadata": "Not a dict!"})
        _ = AIQDocument.from_dict(data)
        data["metadata"] = {"title": "Valid Dictionary"}
        data.update({"document_id": 1234})
        _ = AIQDocument.from_dict(data)
