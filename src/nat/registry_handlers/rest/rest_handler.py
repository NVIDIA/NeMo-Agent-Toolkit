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

import base64
import logging
import os
import shutil
import subprocess
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx

from nat.registry_handlers.registry_handler_base import AbstractRegistryHandler
from nat.registry_handlers.schemas.headers import RequestHeaders
from nat.registry_handlers.schemas.package import PackageNameVersionList
from nat.registry_handlers.schemas.publish import Artifact
from nat.registry_handlers.schemas.publish import PublishResponse
from nat.registry_handlers.schemas.pull import PullRequestPackages
from nat.registry_handlers.schemas.pull import PullResponse
from nat.registry_handlers.schemas.remove import RemoveResponse
from nat.registry_handlers.schemas.search import SearchQuery
from nat.registry_handlers.schemas.search import SearchResponse
from nat.registry_handlers.schemas.status import ActionEnum
from nat.registry_handlers.schemas.status import StatusEnum

logger = logging.getLogger(__name__)


class RestRegistryHandler(AbstractRegistryHandler):
    """A registry handler for interactions with a remote REST registry."""

    def __init__(  # pylint: disable=R0917
            self,
            endpoint: str,
            token: str,
            timeout: int = 30,
            publish_route: str = "",
            pull_route: str = "",
            search_route: str = "",
            remove_route: str = ""):
        super().__init__()
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._publish_route = publish_route.strip("/")
        self._pull_route = pull_route.strip("/")
        self._search_route = search_route.strip("/")
        self._remove_route = remove_route.strip("/")
        self._headers = RequestHeaders(Authorization=f"Bearer: {token}").model_dump(by_alias=True)

    @asynccontextmanager
    async def publish(self, artifact: Artifact) -> AsyncGenerator[PublishResponse]:
        """Publishes a NAT artifact to a remote REST registry.

        Args:
            artifact (Artifact): An artifact that contain NAT plugin wheel and it's corrosponding discovery
            metadata.

        Yields:
            Iterator[AsyncGenerator[PublishResponse]]: A response message that includes a completion status message.
        """

        try:

            async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
                response = await client.post(f"{self._endpoint}/{self._publish_route}",
                                             content=artifact.artifact.model_dump_json())
                response.raise_for_status()
                response_dict = response.json()

            validated_publish_response = PublishResponse(**response_dict)

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

    @asynccontextmanager
    async def pull(self, packages: PullRequestPackages) -> AsyncGenerator[PullResponse]:
        """Download and install NAT artifacts from a remote REST registry.

        Args:
            packages (PullRequestPackages): Parameters used to pull the NAT artifact.

        Yields:
            Iterator[AsyncGenerator[PullResponse]]: A response message that includes a the pulled packages and a
                completion status message.
        """

        tmp_dir = "./.tmp/nat-pull"

        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
                packages_json = packages.model_dump_json()
                response = await client.post(f"{self._endpoint}/{self._pull_route}", content=packages_json)
                response.raise_for_status()

            response_dict = response.json()
            validated_pull_response = PullResponse(**response_dict)

            if (validated_pull_response.status.status == StatusEnum.ERROR):
                logger.error(validated_pull_response.status.message)
                raise ValueError(f"Server error: {validated_pull_response.status.message}")

            if (not os.path.exists(tmp_dir)):
                os.mkdir(tmp_dir)

            whl_paths = []

            for package in validated_pull_response.packages:
                whl_bytes = base64.b64decode(package.whl)
                whl_path = os.path.join(tmp_dir, package.whl_name)

                with open(whl_path, "wb") as f:
                    f.write(whl_bytes)

                whl_paths.append(whl_path)

            cmd = ["uv", "pip", "install"]
            cmd.extend(whl_paths)
            result = subprocess.run(cmd, check=True)
            result.check_returncode()

            if (os.path.exists(tmp_dir)):
                shutil.rmtree(tmp_dir)

            yield validated_pull_response

        except Exception as e:
            msg = f"Error occured when installing packages: {e}"
            logger.error(msg)
            if (os.path.exists(tmp_dir)):
                shutil.rmtree(tmp_dir)

            validated_pull_response = PullResponse(status={
                "status": StatusEnum.ERROR, "message": msg, "action": ActionEnum.PULL
            })
            logger.exception(validated_pull_response.status.message, exc_info=True)

            yield validated_pull_response

        finally:
            logger.info("Execution complete.")

    @asynccontextmanager
    async def search(self, query: SearchQuery) -> AsyncGenerator[SearchResponse]:
        """Searches a remote REST registry for relevant NAT components.

        Args:
            query (SearchQuery): Parameters of the search to be performed.

        Yields:
            Iterator[AsyncGenerator[SearchResponse]]: A response message that includes search
                parameters and a completion status message.
        """

        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
                query_json = query.model_dump_json()
                response = await client.post(url=f"{self._endpoint}/{self._search_route}", content=query_json)
                response.raise_for_status()

            response_dict = response.json()
            validated_search_response = SearchResponse(**response_dict)

            yield validated_search_response

        except Exception as e:
            msg = f"Error searching for artifacts: {e}"
            validated_search_response = SearchResponse(params=query,
                                                       status={
                                                           "status": StatusEnum.ERROR,
                                                           "message": msg,
                                                           "action": ActionEnum.SEARCH
                                                       })
            logger.exception(validated_search_response.status.message, exc_info=True)

            yield validated_search_response

        finally:
            logger.info("Execution complete.")

    @asynccontextmanager
    async def remove(self, packages: PackageNameVersionList) -> AsyncGenerator[RemoveResponse]:
        """Removes packages from a remote REST registry.

        Args:
            packages (PackageNameVersionList): The list of packages to remove.

        Yields:
            Iterator[AsyncGenerator[RemoveResponse]]: A response message that includes the packages and a
                completion status message.
        """

        try:
            async with httpx.AsyncClient(headers=self._headers, timeout=self._timeout) as client:
                remove_json = packages.model_dump_json()
                response = await client.post(url=f"{self._endpoint}/{self._remove_route}", content=remove_json)
                response.raise_for_status()

            response_dict = response.json()
            validated_remove_response = RemoveResponse(**response_dict)

            yield validated_remove_response

        except Exception as e:
            msg = f"Error removing artifacts: {e}"
            validated_remove_response = RemoveResponse(status={
                "status": StatusEnum.ERROR, "message": msg, "action": ActionEnum.REMOVE
            })
            logger.exception(validated_remove_response.status.message, exc_info=True)

            yield validated_remove_response

        finally:
            logger.info("Execution complete.")
