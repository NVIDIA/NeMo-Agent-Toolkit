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

import logging
from enum import Enum

from pydantic import BaseModel

from aiq.data_models.component import ComponentEnum
from aiq.registry_handlers.schemas.status import StatusMessage

logger = logging.getLogger(__name__)


class SearchFields(str, Enum):
    ALL = "all"
    PACKAGE = "package"
    VERSION = "version"
    COMPONENT_NAME = "component_name"
    DESCRIPTION = "description"
    DEVELOPER_NOTES = "developer_notes"


class VisualizeFields(str, Enum):
    PACKAGE = "package"
    VERSION = "version"
    COMPONENT_TYPE = "component_type"
    COMPONENT_NAME = "component_name"
    DESCRIPTION = "description"


class SearchQuery(BaseModel):
    """Represents the search criteria that will be used to discover useful AIQ Toolkit components.

    Args:
        query (str): A query string used to find useful AIQ Toolkit components.
        fields (list[SearchFields]): The list of fields used when applying the query string.
        component_types (list[AIQComponentEnum]): AIQ Toolkit components types to filter search results.
        top_k (int): Specifies the number of search results to provide.
    """

    query: str = "*"
    fields: list[SearchFields] = [SearchFields.ALL]
    component_types: list[ComponentEnum]
    top_k: int = 10


class SearchResponseItem(BaseModel):
    """Represents an individual item in the search response, including elements of it's discovery metadata.

    Args:
        package (str): The name of the AIQ Toolkit package that includes the component.
        version (str): The version of the AIQ Toolkit package that includes the component.
        component_type (AIQComponentEnum): Type of AIQ Toolkit component this item represents.
        description (str): A description of this AIQ Toolkit component.
        developer_notes (str): Additional details that would help a developer use this component.
    """

    package: str
    version: str
    component_type: ComponentEnum
    component_name: str
    description: str
    developer_notes: str


class SearchResponse(BaseModel):
    """Represents a data model of the expected search response.

    Args:
        results (list[SearchResponseItem]): A list of results that matched the search criteria.
        params (SearchQuery): The search criterial that produced these search results.
        status (StatusMessage): Provides metadata describing the success or errors that occurred when making the search
        request.
    """

    results: list[SearchResponseItem] = []
    params: SearchQuery
    status: StatusMessage
