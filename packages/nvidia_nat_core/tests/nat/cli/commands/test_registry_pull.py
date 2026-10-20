# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from unittest.mock import patch

from nat.cli.commands.registry.pull import pull_artifact
from nat.registry_handlers.schemas.package import PackageNameVersion
from nat.registry_handlers.schemas.pull import PullPackageWhl
from nat.registry_handlers.schemas.pull import PullRequestPackages


async def _pull(packages: list[str]) -> PullRequestPackages:
    """Run `pull_artifact` against a stub registry handler and return the request it was given."""

    pulled = []

    class _Handler:

        @asynccontextmanager
        async def pull(self, packages: PullRequestPackages) -> AsyncIterator[None]:
            pulled.append(packages)
            yield

    @asynccontextmanager
    async def _build_fn(config) -> AsyncIterator[_Handler]:
        yield _Handler()

    registry = MagicMock()
    registry.get_registry_handler.return_value.build_fn = _build_fn

    with patch("nat.cli.type_registry.GlobalTypeRegistry.get", return_value=registry):
        await pull_artifact(MagicMock(), packages)

    assert len(pulled) == 1
    return pulled[0]


async def test_pull_artifact_keeps_pinned_version():
    request = await _pull(["nat_simple==1.2.0", "nat_other"])

    assert request.packages == [
        PackageNameVersion(name="nat_simple", version="1.2.0"),
        PackageNameVersion(name="nat_other", version=None),
    ]


async def test_pull_artifact_sends_wheel_path():
    request = await _pull(["dist/nat_simple-0.1.0-py3-none-any.whl"])

    assert request.packages == [PullPackageWhl(whl_path="dist/nat_simple-0.1.0-py3-none-any.whl")]
