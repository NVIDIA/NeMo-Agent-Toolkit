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

import pytest

from nat.plugins.mysql.mysql_object_store import MySQLObjectStore
from nat.plugins.mysql.object_store import MySQLObjectStoreClientConfig
from nat.plugins.mysql.object_store import mysql_object_store_client


@pytest.fixture(name="no_connect")
def no_connect_fixture(monkeypatch):

    async def _aenter(self):
        return self

    async def _aexit(self, *args):
        return None

    monkeypatch.setattr(MySQLObjectStore, "__aenter__", _aenter)
    monkeypatch.setattr(MySQLObjectStore, "__aexit__", _aexit)


@pytest.mark.usefixtures("no_connect")
async def test_mysql_object_store_without_credentials():
    """Username and password are optional in the config and must reach the store as None."""
    config = MySQLObjectStoreClientConfig(bucket_name="test_bucket", username=None, password=None)

    async with mysql_object_store_client(config, None) as store:
        assert isinstance(store, MySQLObjectStore)
        assert store._username is None
        assert store._password is None


@pytest.mark.usefixtures("no_connect")
async def test_mysql_object_store_passes_plain_password():
    config = MySQLObjectStoreClientConfig(bucket_name="test_bucket", username="nat", password="secret")

    async with mysql_object_store_client(config, None) as store:
        assert store._username == "nat"
        assert store._password == "secret"


def test_mysql_config_descriptions_name_env_vars():
    fields = MySQLObjectStoreClientConfig.model_fields

    assert MySQLObjectStoreClientConfig.HOST_ENV in fields["host"].description
    assert MySQLObjectStoreClientConfig.PORT_ENV in fields["port"].description
    assert MySQLObjectStoreClientConfig.PASSWORD_ENV in fields["password"].description
    assert "{" not in fields["host"].description
