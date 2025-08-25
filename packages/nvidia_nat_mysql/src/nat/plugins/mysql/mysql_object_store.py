# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re

import aiomysql
from aiomysql.pool import Pool

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.utils.type_utils import override

logger = logging.getLogger(__name__)


class MySQLObjectStore(ObjectStore):
    """
    Implementation of ObjectStore that stores objects in a MySQL database.
    """

    def __init__(self, *, bucket_name: str, host: str, port: int, username: str | None, password: str | None):
        super().__init__()

        if not re.fullmatch(r"[A-Za-z0-9_-]+", bucket_name):
            raise ValueError("bucket_name must match [A-Za-z0-9_-]+")

        self._bucket_name = bucket_name
        self._host = host
        self._port = port
        self._username = username
        self._password = password

        self._conn_pool: Pool | None = None

    @property
    def _schema(self) -> str:
        return f"`bucket_{self._bucket_name}`"

    async def __aenter__(self) -> "MySQLObjectStore":

        if self._conn_pool is not None:
            raise RuntimeError("Connection already established")

        self._conn_pool = await aiomysql.create_pool(
            host=self._host,
            port=self._port,
            user=self._username,
            password=self._password,
            autocommit=False,  # disable autocommit for transactions
        )
        assert self._conn_pool is not None

        logger.info("Created connection pool for %s at %s:%s", self._bucket_name, self._host, self._port)

        async with self._conn_pool.acquire() as conn:
            async with conn.cursor() as cur:

                # Suppress MySQL "IF NOT EXISTS" notes that surface as warnings in the driver
                await cur.execute("SET sql_notes = 0;")
                try:
                    # Create schema (database) if doesn't exist
                    await cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema} DEFAULT CHARACTER SET utf8mb4;")
                    await cur.execute(f"USE {self._schema};")

                    # Create metadata table_schema
                    await cur.execute("""
                    CREATE TABLE IF NOT EXISTS object_meta (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        path VARCHAR(768) NOT NULL UNIQUE,
                        size BIGINT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ) ENGINE=InnoDB;
                    """)

                    # Create blob data table
                    await cur.execute("""
                    CREATE TABLE IF NOT EXISTS object_data (
                        id INT PRIMARY KEY,
                        data LONGBLOB NOT NULL,
                        FOREIGN KEY (id) REFERENCES object_meta(id) ON DELETE CASCADE
                    ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC;
                    """)
                finally:
                    await cur.execute("SET sql_notes = 1;")

            await conn.commit()

        logger.info("Created schema and tables for %s at %s:%s", self._bucket_name, self._host, self._port)

        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:

        if not self._conn_pool:
            raise RuntimeError("Connection not established")

        # Trigger the non-async close method then wait for the pool to close
        self._conn_pool.close()

        await self._conn_pool.wait_closed()

        self._conn_pool = None

    @override
    async def put_object(self, key: str, item: ObjectStoreItem):

        if not self._conn_pool:
            raise RuntimeError("Connection not established")

        async with self._conn_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"USE {self._schema};")
                try:
                    await cur.execute("START TRANSACTION;")
                    await cur.execute("INSERT IGNORE INTO object_meta (path, size) VALUES (%s, %s)",
                                      (key, len(item.data)))
                    if cur.rowcount == 0:
                        raise KeyAlreadyExistsError(
                            key=key, additional_message=f"MySQL table {self._bucket_name} already has key {key}")
                    await cur.execute("SELECT id FROM object_meta WHERE path=%s FOR UPDATE;", (key, ))
                    (obj_id, ) = await cur.fetchone()
                    blob = item.model_dump_json()
                    await cur.execute("INSERT INTO object_data (id, data) VALUES (%s, %s)", (obj_id, blob))
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise

    @override
    async def upsert_object(self, key: str, item: ObjectStoreItem):

        if not self._conn_pool:
            raise RuntimeError("Connection not established")

        async with self._conn_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"USE {self._schema};")
                try:
                    await cur.execute("START TRANSACTION;")
                    await cur.execute(
                        """
                        INSERT INTO object_meta (path, size)
                        VALUES (%s, %s) AS new
                        ON DUPLICATE KEY UPDATE size=new.size, created_at=CURRENT_TIMESTAMP
                        """, (key, len(item.data)))
                    await cur.execute("SELECT id FROM object_meta WHERE path=%s FOR UPDATE;", (key, ))
                    (obj_id, ) = await cur.fetchone()

                    blob = item.model_dump_json()
                    await cur.execute("REPLACE INTO object_data (id, data) VALUES (%s, %s)", (obj_id, blob))
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise

    @override
    async def get_object(self, key: str) -> ObjectStoreItem:

        if not self._conn_pool:
            raise RuntimeError("Connection not established")

        async with self._conn_pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"USE {self._schema};")
                await cur.execute(
                    """
                    SELECT d.data
                    FROM object_data d
                    JOIN object_meta m USING(id)
                    WHERE m.path=%s
                """, (key, ))
                row = await cur.fetchone()
                if not row:
                    raise NoSuchKeyError(key=key,
                                         additional_message=f"MySQL table {self._bucket_name} does not have key {key}")
                return ObjectStoreItem.model_validate_json(row[0].decode("utf-8"))

    @override
    async def delete_object(self, key: str):

        if not self._conn_pool:
            raise RuntimeError("Connection not established")

        async with self._conn_pool.acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(f"USE {self._schema};")
                    await cur.execute(
                        """
                        DELETE m, d
                        FROM object_meta m
                        JOIN object_data d USING(id)
                        WHERE m.path=%s
                    """, (key, ))

                    if cur.rowcount == 0:
                        raise NoSuchKeyError(
                            key=key, additional_message=f"MySQL table {self._bucket_name} does not have key {key}")

                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
