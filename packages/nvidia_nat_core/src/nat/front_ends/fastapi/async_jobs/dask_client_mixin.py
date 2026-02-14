# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from abc import ABC

if typing.TYPE_CHECKING:
    from dask.distributed import Client as DaskClient


class DaskClientMixin(ABC):

    @property
    def dask_client(self) -> "DaskClient":
        """
        Lazily initializes and returns a Dask Client connected to the specified scheduler address.
        Requires that the inheriting class has a `_scheduler_address` attribute.
        """

        if getattr(self, "_dask_client", None) is None:
            from dask.distributed import Client
            self._dask_client = Client(self._scheduler_address, asynchronous=False)

        return self._dask_client
