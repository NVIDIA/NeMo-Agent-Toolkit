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

if typing.TYPE_CHECKING:
    from dask.distributed import Client as DaskClient


def wait_job(dask_client: "DaskClient", job_id: str, timeout: int = 60) -> typing.Any:
    """Helper to wait for a job to complete."""
    from dask.distributed import Variable

    var = Variable(name=job_id, client=dask_client)
    future = var.get(timeout=5)
    results = future.result(timeout=timeout)

    return results
