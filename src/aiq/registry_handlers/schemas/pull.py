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

from pydantic import BaseModel

from aiq.registry_handlers.schemas.package import PackageNameVersion
from aiq.registry_handlers.schemas.status import StatusMessage

logger = logging.getLogger(__name__)


class PulledPackage(BaseModel):
    """Represents a data model of a pulled package containing the package wheel and its name.

    Args:
        whl (str): Base64 encoded string of the AgentIQ python package wheel (.whl).
        whl_name (str): A string representing the wheel filename.
    """

    whl: str
    whl_name: str


class PullResponse(BaseModel):
    """
    Represents a data model of the expected respones from a AgentIQ pull request, including detailed status information.

    Args:
        packages (list[PulledPackage]): A list of pulled packages included in the pull request.
        status (StatusMessage): Provides metadata describing the success or errors that occurred when making to pull in
        a package.
    """

    packages: list[PulledPackage] = []
    status: StatusMessage


class PullPackageWhl(BaseModel):
    """Local path to wheel (.whl) file.

    Args:
        whl_path (str): The local path the wheel (.whl) file.
    """

    whl_path: str


class PullRequestPackage(BaseModel):
    """Represents all data for a single package needed to download an install its components.

    Args:
        package (typing.Union[PackageNameVersion, PullPackageWhl]): Attributes of a single package necessary
            to download and install its components.
    """

    package: PackageNameVersion | PullPackageWhl


class PullRequestPackages(BaseModel):
    """Represents a list of all packages th download and install in the local AgentIQ environment.

    Args:
        packages (list[typing.Union[PackageNameVersion, PullPackageWhl]]): A list of packages that can be
            downloaded and installed in the local AgentIQ environment.
    """

    packages: list[PackageNameVersion | PullPackageWhl]
