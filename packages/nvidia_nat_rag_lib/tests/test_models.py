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

import pytest
from nvidia_rag.rag_server.response_generator import Citations

from nat.plugins.rag_lib.models import RAGGenerateResult
from nat.plugins.rag_lib.models import RAGSearchResult

# Rebuild models to resolve forward references to Citations
RAGSearchResult.model_rebuild()
RAGGenerateResult.model_rebuild()


class TestRAGSearchResult:
    """Tests for RAGSearchResult model."""

    @pytest.fixture
    def citations(self) -> Citations:
        """Create Citations object."""
        return Citations(total_results=2, results=[])

    def test_creation(self, citations: Citations) -> None:
        """Test RAGSearchResult can be created with citations."""
        result = RAGSearchResult(citations=citations)
        assert result.citations is citations

    def test_str_returns_json(self, citations: Citations) -> None:
        """Test __str__ returns JSON from citations.model_dump_json()."""
        result = RAGSearchResult(citations=citations)
        output = str(result)

        parsed = json.loads(output)
        assert parsed["total_results"] == 2


class TestRAGGenerateResult:
    """Tests for RAGGenerateResult model."""

    @pytest.fixture
    def citations(self) -> Citations:
        """Create Citations object."""
        return Citations(total_results=1, results=[])

    def test_creation_with_answer_only(self) -> None:
        """Test RAGGenerateResult can be created with just an answer."""
        result = RAGGenerateResult(answer="This is the answer.")
        assert result.answer == "This is the answer."
        assert result.citations is None

    def test_creation_with_citations(self, citations: Citations) -> None:
        """Test RAGGenerateResult can be created with answer and citations."""
        result = RAGGenerateResult(answer="Answer with sources.", citations=citations)
        assert result.answer == "Answer with sources."
        assert result.citations is citations

    def test_str_without_citations(self) -> None:
        """Test __str__ excludes citations when None."""
        result = RAGGenerateResult(answer="Just an answer.")
        output = str(result)

        parsed = json.loads(output)
        assert parsed["answer"] == "Just an answer."
        assert "citations" not in parsed

    def test_str_with_citations(self, citations: Citations) -> None:
        """Test __str__ includes citations when present."""
        result = RAGGenerateResult(answer="Answer.", citations=citations)
        output = str(result)

        parsed = json.loads(output)
        assert parsed["answer"] == "Answer."
        assert "citations" in parsed
