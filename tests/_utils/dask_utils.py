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

import os
import typing


async def await_job(job_id: str, timeout: int = 60) -> typing.Any:
    """Helper to await a job completion."""
    from dask.distributed import Client as DaskClient
    from dask.distributed import Variable

    client = await DaskClient(address=os.environ["NAT_DASK_SCHEDULER_ADDRESS"], asynchronous=True)
    results = None

    try:
        var = Variable(name=job_id, client=client)
        future = await var.get(timeout=5)
        results = await future.result(timeout=timeout)

    finally:
        await client.close()

    return results
