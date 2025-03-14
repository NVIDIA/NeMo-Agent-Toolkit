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


class RemoveResponse(BaseModel):
    """Represents a data model for the expected response from a remove request, including packages and status metadata.

    Args:
        packages (list[PackageNameVersion]): A list of packages that are to be removed from a remote registry.
        status (StatusMessage): Provides metadata describing the success or errors that occurred when making a remove
        request.
    """

    packages: list[PackageNameVersion] = []
    status: StatusMessage
