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

from contextlib import asynccontextmanager

import pytest

from nat.builder.workflow_builder import WorkflowBuilder
from nat.plugins.redis.object_store import RedisObjectStoreClientConfig
from nat.test.object_store_tests import ObjectStoreTests

# NOTE: This test requires a local Redis server to be running.
# To launch a local server using docker, run the following command:
# docker run --rm -ti --name test-redis -p 6379:6379 redis:7-alpine


@pytest.fixture(name="redis_server", scope="module")
def fixture_redis_server(fail_missing: bool):
    """Fixture to safely skip redis based tests if redis is not running"""
    try:
        import redis
        client = redis.Redis(host="localhost", port=6379, db=0)
        if not client.ping():
            raise RuntimeError("Failed to connect to Redis")
        yield
    except ImportError:
        if fail_missing:
            raise
        pytest.skip("redis not installed, skipping redis tests")
    except Exception as e:
        if fail_missing:
            raise
        pytest.skip(f"Error connecting to Redis server: {e}, skipping redis tests")


@pytest.mark.integration
@pytest.mark.usefixtures("redis_server")
class TestRedisObjectStore(ObjectStoreTests):

    @asynccontextmanager
    async def _get_store(self):
        async with WorkflowBuilder() as builder:
            await builder.add_object_store(
                "object_store_name",
                RedisObjectStoreClientConfig(bucket_name="test", host="localhost", port=6379, db=0),
            )

            yield await builder.get_object_store_client("object_store_name")
