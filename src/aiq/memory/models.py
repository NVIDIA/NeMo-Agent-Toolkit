# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import typing

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class MemoryItem(BaseModel):
    """
    Represents a single memory item consisting of structured content and associated metadata.

    Attributes
    ----------
    conversation : list[dict[str, str]]
        A list of dictionaries, each containing string key-value pairs.
    user_id : str
        Unique identifier for this MemoryItem's user.
    tags : list[str]
        A list of strings representing tags attached to the item.
    metadata : dict[str, typing.Any]
        Metadata providing context and utility for management operations.
    memory : str or None
        Optional memory string. Helpful when returning a memory.
    """
    # yapf: disable
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "conversation": [
                        {
                            "role": "user",
                            "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."
                        },
                        {
                            "role": "assistant",
                            "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy."
                        }
                    ],
                    "user_id": "user_abc",
                    "tags": ["diet", "allergy"],
                    "metadata": {
                        "key_value_pairs": {
                            "type": "profile",
                            "relevance": "high"
                        }
                    }
                },
                {
                    "memory": "User prefers expensive hotels and is vegan.",
                    "user_id": "user_abc",
                    "tags": ["hotel", "restaurant"]
                }
            ]
        },
        # Allow population of models from arbitrary types (e.g., ORM objects)
        arbitrary_types_allowed=True,
        # Enable aliasing if needed
        populate_by_name=True
    )
    # yapf: enable
    conversation: list[dict[str, str]] | None = Field(
        description="List of conversation messages. Each message must have a \"role\" "
        "key (user or assistant. It must also have a \"content\" key.",
        default=None)
    tags: list[str] = Field(default_factory=list, description="List of tags applied to the item.")
    metadata: dict[str, typing.Any] = Field(description="Metadata about the memory item.", default={})
    user_id: str = Field(description="The user's ID.")
    memory: str | None = Field(default=None)


class SearchMemoryInput(BaseModel):
    """
    Represents a search memory input structure.
    """
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What is the user's preferred programming language?",
            "top_k": 1,
            "user_id": "user_abc",
        }
    })

    query: str = Field(description="Search query for which to retrieve memory.")  # noqa: E501
    top_k: int = Field(description="Maximum number of memories to return")
    user_id: str = Field(description="ID of the user to search for.")


class DeleteMemoryInput(BaseModel):
    """
    Represents a delete memory input structure.
    """
    model_config = ConfigDict(json_schema_extra={"example": {"user_id": "user_abc", }})

    user_id: str = Field(description="ID of the user to delete memory for. Careful when using "
                         "this tool; make sure you use the "
                         "username present in the conversation.")
