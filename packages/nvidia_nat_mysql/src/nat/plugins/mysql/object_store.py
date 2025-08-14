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
from typing import ClassVar

from pydantic import Field

from nat.builder.builder import Builder
from nat.cli.register_workflow import register_object_store
from nat.data_models.object_store import ObjectStoreBaseConfig


class MySQLObjectStoreClientConfig(ObjectStoreBaseConfig, name="mysql"):
    """
    Object store that stores objects in a MySQL database.
    """

    DEFAULT_HOST: ClassVar[str] = "localhost"
    DEFAULT_PORT: ClassVar[int] = 3306

    HOST_ENV: ClassVar[str] = "NAT_MYSQL_OBJECT_STORE_HOST"
    PORT_ENV: ClassVar[str] = "NAT_MYSQL_OBJECT_STORE_PORT"
    USERNAME_ENV: ClassVar[str] = "NAT_MYSQL_OBJECT_STORE_USERNAME"
    PASSWORD_ENV: ClassVar[str] = "NAT_MYSQL_OBJECT_STORE_PASSWORD"

    bucket_name: str = Field(description="The name of the bucket to use for the object store")
    host: str = Field(
        default=os.environ.get(HOST_ENV, DEFAULT_HOST),
        description="The host of the MySQL server"
        " (uses {HOST_ENV} if unspecified; falls back to {DEFAULT_HOST})",
    )
    port: int = Field(
        default=int(os.environ.get(PORT_ENV, DEFAULT_PORT)),
        description="The port of the MySQL server"
        " (uses {PORT_ENV} if unspecified; falls back to {DEFAULT_PORT})",
    )
    username: str | None = Field(
        default=os.environ.get(USERNAME_ENV),
        description=f"The username used to connect to the MySQL server (uses {USERNAME_ENV} if unspecifed)",
    )
    password: str | None = Field(
        default=os.environ.get(PASSWORD_ENV),
        description="The password used to connect to the MySQL server (uses {PASSWORD_ENV} if unspecifed)",
    )


@register_object_store(config_type=MySQLObjectStoreClientConfig)
async def mysql_object_store_client(config: MySQLObjectStoreClientConfig, builder: Builder):

    from .mysql_object_store import MySQLObjectStore

    async with MySQLObjectStore(config) as store:
        yield store
