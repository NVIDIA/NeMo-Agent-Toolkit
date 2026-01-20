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

from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from nvidia_rag.rag_server.response_generator import Citations


class RAGResultBase(BaseModel):
    """Base model for RAG tool results."""

    model_config = {"arbitrary_types_allowed": True}


class RAGSearchResult(RAGResultBase):
    """RAG search result."""

    citations: "Citations"

    def __str__(self) -> str:
        return self.citations.model_dump_json()


class RAGGenerateResult(RAGResultBase):
    """RAG generation result."""

    answer: str
    citations: "Citations | None" = None

    def __str__(self) -> str:
        return self.model_dump_json(exclude_none=True)
