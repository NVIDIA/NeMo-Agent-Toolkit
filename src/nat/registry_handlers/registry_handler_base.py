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

from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum

from nat.data_models.component import ComponentEnum
from nat.data_models.discovery_metadata import DiscoveryMetadata
from nat.registry_handlers.schemas.package import PackageNameVersionList
from nat.registry_handlers.schemas.publish import Artifact
from nat.registry_handlers.schemas.publish import PublishResponse
from nat.registry_handlers.schemas.pull import PullRequestPackages
from nat.registry_handlers.schemas.pull import PullResponse
from nat.registry_handlers.schemas.remove import RemoveResponse
from nat.registry_handlers.schemas.search import SearchQuery
from nat.registry_handlers.schemas.search import SearchResponse
from nat.registry_handlers.schemas.search import VisualizeFields


class AbstractRegistryHandler(ABC):
    """Base class outlining the interfaces for remote AIQ Toolkit registry interactions."""

    def __init__(self):
        self._discovery_metadata: dict[ComponentEnum, list[dict | DiscoveryMetadata]] = {}
        self._aiq_artifact: Artifact | None = None
        self._whl_bytes: bytes
        self._whl_path: str
        self._whl_base64: str

    @abstractmethod
    @asynccontextmanager
    async def publish(self, artifact: Artifact) -> AsyncGenerator[PublishResponse]:
        """Publishes an AIQ Toolkit artifact to a remote registry.

        Args:
            artifact (AIQArtifact): An artifact that contain AIQ Toolkit plugin wheel and it's corrosponding discovery
            metadata.

        Yields:
            Iterator[AsyncGenerator[PublishResponse, None]]: A response message that includes a completion status
            message.
        """

        pass

    @abstractmethod
    @asynccontextmanager
    async def pull(self, packages: PullRequestPackages) -> AsyncGenerator[PullResponse]:
        """Download and install AIQ Toolkit artifacts from a remote registry.

        Args:
            packages (PullRequestPackages): Parameters used to pull the AIQ Toolkit artifact.

        Yields:
            Iterator[AsyncGenerator[PullResponse]]: A response message that includes a the pulled packages and a
                completion status message.
        """

        pass

    @abstractmethod
    @asynccontextmanager
    async def search(self, query: SearchQuery) -> AsyncGenerator[SearchResponse]:
        """Searches the local aiq registry for relevant AIQ Toolkit components.

        Args:
            query (SearchQuery): Parameters of the search to be performed.

        Yields:
            Iterator[AsyncGenerator[SearchResponse]]: A response message that includes search
                parameters and a completion status message.
        """

        pass

    @abstractmethod
    @asynccontextmanager
    async def remove(self, packages: PackageNameVersionList) -> AsyncGenerator[RemoveResponse]:
        """Removes packages from a remote registry.

        Args:
            packages (PackageNameVersionList): The list of packages to remove.

        Yields:
            Iterator[AsyncGenerator[RemoveResponse]]: A response message that includes the packages and a
                completion status message.
        """

        pass

    @staticmethod
    def visualize_search_results(search_response: SearchResponse, pager: bool = True) -> None:
        """Visualze search results in a system terminal.

        Args:
            search_response (SearchResponse): A response message that includes search parameters and a completion status
            message.

            pager (bool, optional): Include an pagable terminal interface for large search results. Defaults to False.
        """

        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        table = Table(title="AIQ Toolkit Search Results", padding=(0, 1), show_lines=True)
        for column in VisualizeFields:
            table.add_column(column.value)

        for result in search_response.results:
            row = []
            for column in VisualizeFields:
                value = getattr(result, column.value)
                if isinstance(value, Enum):
                    value = value.value
                text = Text(value, overflow="fold")
                row.append(text)
            table.add_row(*row, style='bright_green')

        console = Console()

        if (pager):
            with console.pager():
                console.print(table)
        else:
            console.print(table)

    @staticmethod
    def save_search_results(search_response: SearchResponse, save_path: str) -> None:
        """Save search results to a local json file.

        Args:
            search_response (SearchResponse): A response message that includes search parameters and a completion status
            message.

            save_path (str): The path to save the json search results.
        """

        search_response_str = search_response.model_dump_json(indent=4)

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(search_response_str)
