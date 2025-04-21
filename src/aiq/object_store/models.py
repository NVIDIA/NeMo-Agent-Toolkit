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

from pydantic import BaseModel
from pydantic import Field


class ObjectStoreItem(BaseModel):
    """
    Represents an object store item consisting of bytes and associated metadata.

    Attributes
    ----------
    data : bytes
        The data to store in the object store.
    content_type : str | None
        The content type of the data.
    metadata : dict[str, str] | None
        Metadata providing context and utility for management operations.
    """

    data: bytes = Field(description="The data to store in the object store.")
    content_type: str | None = Field(description="The content type of the data.", default=None)
    metadata: dict[str, str] | None = Field(description="The metadata of the data.", default=None)
