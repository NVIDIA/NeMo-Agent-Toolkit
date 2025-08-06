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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_object_store
from aiq.data_models.object_store import ObjectStoreBaseConfig


class MySQLObjectStoreClientConfig(ObjectStoreBaseConfig, name="mysql"):
    """
    Object store that stores objects in a MySQL database.
    """

    DEFAULT_ENDPOINT: ClassVar[str] = "127.0.0.1:3306"

    ENDPOINT_ENV: ClassVar[str] = "AIQ_MYSQL_OBJECT_STORE_ENDPOINT"
    USER_ENV: ClassVar[str] = "AIQ_MYSQL_OBJECT_STORE_USER"
    PASSWORD_ENV: ClassVar[str] = "AIQ_MYSQL_OBJECT_STORE_PASSWORD"

    bucket_name: str = Field(description="The name of the bucket to use for the object store")
    endpoint_url: str = Field(
        default=os.environ.get(ENDPOINT_ENV, DEFAULT_ENDPOINT),
        description="The host and port of the MySQL server"
        " (uses {ENDPOINT_ENV} if unspecified; falls back to {DEFAULT_ENDPOINT})",
    )
    user: str | None = Field(
        default=os.environ.get(USER_ENV),
        description=f"The user used to connect to the MySQL server (uses {USER_ENV} if unspecifed)",
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
