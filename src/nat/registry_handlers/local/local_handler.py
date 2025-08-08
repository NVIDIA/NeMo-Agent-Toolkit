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
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from aiq.registry_handlers.package_utils import build_package_metadata
from aiq.registry_handlers.registry_handler_base import AbstractRegistryHandler
from aiq.registry_handlers.schemas.package import PackageNameVersionList
from aiq.registry_handlers.schemas.publish import Artifact
from aiq.registry_handlers.schemas.publish import PublishResponse
from aiq.registry_handlers.schemas.pull import PullRequestPackages
from aiq.registry_handlers.schemas.pull import PullResponse
from aiq.registry_handlers.schemas.remove import RemoveResponse
from aiq.registry_handlers.schemas.search import SearchFields
from aiq.registry_handlers.schemas.search import SearchQuery
from aiq.registry_handlers.schemas.search import SearchResponse
from aiq.registry_handlers.schemas.status import ActionEnum
from aiq.registry_handlers.schemas.status import StatusEnum

logger = logging.getLogger(__name__)


class LocalRegistryHandler(AbstractRegistryHandler):
    """A registry handler for interactions with a local Python environment."""

    search_fields: list[SearchFields] = [field for field in SearchFields if field != SearchFields.ALL]

    @asynccontextmanager
    async def publish(self, artifact: Artifact) -> AsyncGenerator[PublishResponse]:
        """Publishes an AIQ Toolkit artifact to a local registry.

        Args:
            artifact (AIQArtifact): An artifact that contain AIQ Toolkit plugin wheel and it's corrosponding discovery
            metadata.

        Yields:
            Iterator[AsyncGenerator[PublishResponse]]: A response message that includes a completion status message.
        """

        try:
            validated_remove_response = RemoveResponse(status={
                "status": StatusEnum.ERROR, "message": "Local publish not supported.", "action": ActionEnum.PUBLISH
            })
            yield validated_remove_response
        finally:
            logger.warning(validated_remove_response.status.message)

    @asynccontextmanager
    async def pull(self, packages: PullRequestPackages) -> AsyncGenerator[PullResponse]:
        """Download and install AIQ Toolkit artifacts from a local registry.

        Args:
            packages (PullRequestPackages): Parameters used to pull the AIQ Toolkit artifact.

        Yields:
            Iterator[AsyncGenerator[PullResponse]]: A response message that includes a the pulled packages and a
                completion status message.
        """

        try:
            validated_remove_response = RemoveResponse(status={
                "status": StatusEnum.ERROR, "message": "Local pull not supported.", "action": ActionEnum.PULL
            })

            yield validated_remove_response
        finally:
            logger.warning(validated_remove_response.status.message)

    @asynccontextmanager
    async def search(self, query: SearchQuery) -> AsyncGenerator[SearchResponse]:
        """Searches the local aiq registry for relevant AIQ Toolkit components.

        Args:
            query (SearchQuery): Parameters of the search to be performed.

        Yields:
            Iterator[AsyncGenerator[SearchResponse]]: A response message that includes search
                parameters and a completion status message.
        """

        try:
            results_dict = build_package_metadata(wheel_data=None)
            component_results = []
            query_component_types = [component_type.value for component_type in query.component_types]
            for component_type, components in results_dict.items():
                if component_type in query_component_types:
                    component_results.extend(components)

            if (SearchFields.ALL in query.fields):
                query.fields = self.search_fields

            matched_results = []
            for component_result in component_results:
                for search_field in query.fields:
                    if (query.query in component_result.get(search_field.value, "")):
                        matched_results.append(component_result)
                        break

            if query.top_k > 0:
                top_k = query.top_k
            else:
                top_k = len(matched_results)

            validated_search_response = SearchResponse(results=matched_results[:top_k],
                                                       params=query,
                                                       status={
                                                           "status": StatusEnum.SUCCESS,
                                                           "message": "",
                                                           "action": ActionEnum.SEARCH
                                                       })
            yield validated_search_response

        except Exception as e:
            msg = f"Error searching for artifacts: {e}"
            validated_search_response = SearchResponse(params=query,
                                                       status={
                                                           "status": StatusEnum.SUCCESS,
                                                           "message": msg,
                                                           "action": ActionEnum.SEARCH
                                                       })
            logger.exception(validated_search_response.status.message, exc_info=True)

            yield validated_search_response

        finally:
            logger.info("Execution complete.")

    @asynccontextmanager
    async def remove(self, packages: PackageNameVersionList) -> AsyncGenerator[RemoveResponse]:
        """Uninstall packages from the local Python environment.

        Args:
            packages (PackageNameVersionList): The list of packages to remove.

        Yields:
            Iterator[AsyncGenerator[RemoveResponse]]: A response message that includes the packages and a
                completion status message.
        """

        try:
            for package_name in packages.packages:
                result = subprocess.run(["uv", "pip", "uninstall", package_name.name], check=True)
                result.check_returncode()

            validated_remove_response = RemoveResponse(status={
                "status": StatusEnum.SUCCESS, "message": "", "action": ActionEnum.REMOVE
            })  # type: ignore

            yield validated_remove_response

        except Exception as e:
            msg = f"Error uninstalling artifacts: {e}"
            validated_remove_response = RemoveResponse(status={
                "status": StatusEnum.ERROR, "message": msg, "action": ActionEnum.REMOVE
            })  # type: ignore
            logger.exception(validated_remove_response.status.message, exc_info=True)

            yield validated_remove_response

        finally:
            logger.info("Execution complete.")
