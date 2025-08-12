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

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel
from pydantic import Field

from nat.utils.type_converter import GlobalTypeConverter


class Document(BaseModel):
    """
    Object representing a retrieved document/chunk from a standard NAT Retriever.
    """
    page_content: str = Field(description="Primary content of the document to insert or retrieve")
    metadata: dict[str, Any] = Field(description="Metadata dictionary attached to the Document")
    document_id: str | None = Field(description="Unique ID for the document, if supported by the configured datastore",
                                    default=None)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """
        Deserialize an Document from a dictionary representation.

        Args:
            data (dict): A dictionary containing keys
            'page_content', 'metadata', and optionally 'document_id'.

        Returns:
            MemoryItem: A reconstructed MemoryItem instance.
        """
        return cls(**data)


class RetrieverOutput(BaseModel):
    results: list[Document] = Field(description="A list of retrieved Documents")

    def __len__(self):
        return len(self.results)

    def __str__(self):
        return json.dumps(self.model_dump())


class RetrieverError(Exception):
    pass


def retriever_output_to_dict(obj: RetrieverOutput) -> dict:
    return obj.model_dump()


def retriever_output_to_str(obj: RetrieverOutput) -> str:
    return str(obj)


GlobalTypeConverter.register_converter(retriever_output_to_dict)
GlobalTypeConverter.register_converter(retriever_output_to_str)

# Compatibility aliases with previous releases
AIQDocument = Document
