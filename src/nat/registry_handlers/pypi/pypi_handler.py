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

from nat.data_models.component import ComponentEnum
from nat.registry_handlers.registry_handler_base import AbstractRegistryHandler
from nat.registry_handlers.schemas.package import PackageNameVersionList
from nat.registry_handlers.schemas.publish import Artifact
from nat.registry_handlers.schemas.publish import PublishResponse
from nat.registry_handlers.schemas.pull import PackageNameVersion
from nat.registry_handlers.schemas.pull import PullRequestPackages
from nat.registry_handlers.schemas.pull import PullResponse
from nat.registry_handlers.schemas.remove import RemoveResponse
from nat.registry_handlers.schemas.search import SearchQuery
from nat.registry_handlers.schemas.search import SearchResponse
from nat.registry_handlers.schemas.search import SearchResponseItem
from nat.registry_handlers.schemas.status import ActionEnum
from nat.registry_handlers.schemas.status import StatusEnum

logger = logging.getLogger(__name__)


class PypiRegistryHandler(AbstractRegistryHandler):
    """
    A registry handler for interactions with a remote PyPI registry.

    Built interfacing with this private PyPI server:
    https://github.com/pypiserver/pypiserver
    """

    def __init__(  # pylint: disable=R0917
            self,
            endpoint: str,
            token: str | None = None,
            publish_route: str = "",
            pull_route: str = "",
            search_route: str = ""):
        super().__init__()
        self._endpoint = endpoint.rstrip("/")
        self._token = token
        self._publish_route = publish_route.strip("/")
        self._pull_route = pull_route.strip("/")
        self._search_route = search_route.strip("/")

    @asynccontextmanager
    async def publish(self, artifact: Artifact) -> AsyncGenerator[PublishResponse]:
        """Publishes an AIQ Toolkit artifact to a PyPI remote registry.

        Args:
            artifact (AIQArtifact): An artifact that contain AIQ Toolkit plugin wheel and it's corrosponding discovery
            metadata.

        Yields:
            Iterator[AsyncGenerator[PublishResponse, None]]: A response message that includes a completion status
            message.
        """

        try:
            result = self._upload_to_pypi(wheel_path=artifact.whl_path)
            result.check_returncode()

            validated_publish_response = PublishResponse(status={
                "status": StatusEnum.SUCCESS, "message": "", "action": ActionEnum.PUBLISH
            })

            yield validated_publish_response

        except Exception as e:
            msg = f"Error publishing package: {e}"
            validated_publish_response = PublishResponse(status={
                "status": StatusEnum.ERROR, "message": msg, "action": ActionEnum.PUBLISH
            })
            logger.exception(validated_publish_response.status.message, exc_info=True)

            yield validated_publish_response

        finally:
            logger.info("Execution complete.")

    def _upload_to_pypi(self, wheel_path: str) -> None:

        return subprocess.run(
            ["twine", "upload", "--repository-url", f"{self._endpoint}/{self._publish_route}", f"{wheel_path}"],
            check=True)

    @asynccontextmanager
    async def pull(self, packages: PullRequestPackages) -> AsyncGenerator[PullResponse]:
        """Download and install AIQ Toolkit artifacts from a remote PyPI remote registry.

        Args:
            packages (PullRequestPackages): Parameters used to pull the AIQ Toolkit artifact.

        Yields:
            Iterator[AsyncGenerator[PullResponse, None]]: A response message that includes a the pulled packages and a
                completion status message.
        """

        try:
            versioned_packages = []
            for package in packages.packages:

                if isinstance(package, PackageNameVersion):
                    if (hasattr(package, "version") and package.version is not None):
                        versioned_package = f"{package.name}=={package.version}"
                    else:
                        versioned_package = package.name
                    versioned_packages.append(versioned_package)
                else:
                    versioned_packages.append(package.whl_path)

            versioned_packages_str = " ".join(versioned_packages)

            result = subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--prerelease=allow",
                    "--index-url",
                    f"{self._endpoint}/{self._pull_route}/",
                    versioned_packages_str
                ],  # pylint: disable=W0631
                check=True)

            result.check_returncode()

            validated_pull_response = PullResponse(status={
                "status": StatusEnum.SUCCESS, "message": "", "action": ActionEnum.PULL
            })

            yield validated_pull_response

        except Exception as e:
            msg = f"Error pulling packages: {e}"
            validated_pull_response = PullResponse(status={
                "status": StatusEnum.ERROR, "message": msg, "action": ActionEnum.PULL
            })
            logger.exception(validated_pull_response.status.message, exc_info=True)

            yield validated_pull_response

        finally:
            logger.info("Execution complete.")

    @asynccontextmanager
    async def search(self, query: SearchQuery) -> AsyncGenerator[SearchResponse]:
        """Searches a remote PyPI registry for relevant AIQ Toolkit components.

        Args:
            query (SearchQuery): Parameters of the search to be performed.

        Yields:
            Iterator[AsyncGenerator[SearchResponse]]: A response message that includes search
                parameters and a completion status message.
        """

        try:
            completed_process = subprocess.run(
                ["pip", "search", "--index", f"{self._endpoint}", query.query],  # pylint: disable=W0631
                text=True,
                capture_output=True,
                check=True)
            search_response_list = []
            search_results = completed_process.stdout
            package_results = search_results.split("\n")

            for package_result in package_results:

                # Filter out empty and nested values
                if ((package_result == "") or (package_result[0] == " ")):
                    continue

                package_split = package_result.split(" ")
                package = package_split[0]
                version = package_split[1][1:-1]

                search_resp_item = SearchResponseItem(package=package,
                                                      version=version,
                                                      component_type=ComponentEnum.PACKAGE,
                                                      component_name=package,
                                                      description="",
                                                      developer_notes="")

                if (search_resp_item not in search_response_list):
                    search_response_list.append(search_resp_item)

                    if (len(search_response_list) > query.top_k):
                        break

            validated_search_response = SearchResponse(results=search_response_list,
                                                       params=query,
                                                       status={
                                                           "status": StatusEnum.SUCCESS,
                                                           "message": "",
                                                           "action": ActionEnum.SEARCH
                                                       })

            yield validated_search_response

        except Exception as e:
            msg = f"Error searching for artifacts: {e}"
            logger.exception(msg, exc_info=True)
            validated_search_response = SearchResponse(params=query,
                                                       status={
                                                           "status": StatusEnum.ERROR,
                                                           "message": msg,
                                                           "action": ActionEnum.SEARCH
                                                       })

            yield validated_search_response

        finally:
            logger.info("Execution complete.")

    @asynccontextmanager
    async def remove(self, packages: PackageNameVersionList) -> AsyncGenerator[SearchResponse]:
        """Removes packages from a remote registry.

        Args:
            packages (PackageNameVersionList): The list of packages to remove.

        Yields:
            Iterator[AsyncGenerator[RemoveResponse]]: A response message that includes the packages and a
                completion status message.
        """

        try:
            msg = "PyPI remove not supported."
            validated_remove_response = RemoveResponse(status={
                "status": StatusEnum.ERROR, "message": msg, "action": ActionEnum.REMOVE
            })

            yield validated_remove_response
        finally:
            logger.warning(validated_remove_response.status.message)
