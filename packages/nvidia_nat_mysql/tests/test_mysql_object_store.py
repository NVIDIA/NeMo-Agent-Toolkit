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
from nat.plugins.mysql.object_store import MySQLObjectStoreClientConfig
from nat.test.object_store_tests import ObjectStoreTests

# NOTE: This test requires a MySQL server to be running locally.
# To launch a local server using docker, run the following command:
# docker run --rm -ti --name test-mysql -e MYSQL_ROOT_PASSWORD=my_password -d -p 3306:3306 mysql:9.3


@pytest.fixture(scope='class', autouse=True)
async def _mysql_server(request, mysql_server: dict[str, str | int]):
    request.cls._mysql_server_info = mysql_server


@pytest.mark.integration
@pytest.mark.usefixtures("mysql_server")
class TestMySQLObjectStore(ObjectStoreTests):

    @asynccontextmanager
    async def _get_store(self):
        async with WorkflowBuilder() as builder:
            await builder.add_object_store(
                "object_store_name",
                MySQLObjectStoreClientConfig(**self._mysql_server_info),
            )

            yield await builder.get_object_store_client("object_store_name")
