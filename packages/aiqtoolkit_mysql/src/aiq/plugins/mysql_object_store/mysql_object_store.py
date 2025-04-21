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

import pickle

import aiomysql
import mysql.connector

from aiq.data_models.object_store import KeyAlreadyExistsError
from aiq.data_models.object_store import NoSuchKeyError
from aiq.object_store.interfaces import ObjectStore
from aiq.object_store.models import ObjectStoreItem
from aiq.plugins.mysql_object_store.object_store import MySQLObjectStoreClientConfig


class MySQLObjectStore(ObjectStore):

    def __init__(self, config: MySQLObjectStoreClientConfig):
        self.config = config
        self.conn_pool = None

        conn = mysql.connector.connect(
            host=config.endpoint_url.split(":")[0],
            port=int(config.endpoint_url.split(":")[1]),
            user=config.access_key,
            password=config.secret_key,
        )

        self.schema = f"`bucket_{config.bucket_name}`"

        cur = conn.cursor()
        # Create schema (database) if doesn't exist
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema} DEFAULT CHARACTER SET utf8mb4;")
        cur.execute(f"USE {self.schema};")

        # Create metadata table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS object_meta (
            id INT AUTO_INCREMENT PRIMARY KEY,
            path VARCHAR(768) NOT NULL UNIQUE,
            size BIGINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB;
        """)

        # Create blob data table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS object_data (
            id INT PRIMARY KEY,
            data LONGBLOB NOT NULL,
            FOREIGN KEY (id) REFERENCES object_meta(id) ON DELETE CASCADE
        ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC;
        """)

        conn.commit()
        cur.close()

    async def get_async_conn_pool(self):
        if self.conn_pool is None:
            self.conn_pool = await aiomysql.create_pool(
                host=self.config.endpoint_url.split(":")[0],
                port=int(self.config.endpoint_url.split(":")[1]),
                user=self.config.access_key,
                password=self.config.secret_key,
                autocommit=False,  # disable autocommit for transactions
            )
        return self.conn_pool

    async def put_object(self, key: str, item: ObjectStoreItem):
        async with (await self.get_async_conn_pool()).acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"USE {self.schema};")
                try:
                    await cur.execute("START TRANSACTION;")
                    await cur.execute("INSERT IGNORE INTO object_meta (path, size) VALUES (%s, %s)",
                                      (key, len(item.data)))
                    if cur.rowcount == 0:
                        raise KeyAlreadyExistsError(key=key)
                    await cur.execute("SELECT id FROM object_meta WHERE path=%s FOR UPDATE;", (key, ))
                    (obj_id, ) = await cur.fetchone()

                    blob = pickle.dumps(item)
                    await cur.execute("INSERT INTO object_data (id, data) VALUES (%s, %s)", (obj_id, blob))
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise

    async def upsert_object(self, key: str, item: ObjectStoreItem):
        async with (await self.get_async_conn_pool()).acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"USE {self.schema};")
                try:
                    await cur.execute("START TRANSACTION;")
                    await cur.execute(
                        """
                        INSERT INTO object_meta (path, size)
                        VALUES (%s, %s)
                        ON DUPLICATE KEY UPDATE size=VALUES(size), created_at=CURRENT_TIMESTAMP
                        """, (key, len(item.data)))
                    await cur.execute("SELECT id FROM object_meta WHERE path=%s FOR UPDATE;", (key, ))
                    (obj_id, ) = await cur.fetchone()

                    blob = pickle.dumps(item)
                    await cur.execute("REPLACE INTO object_data (id, data) VALUES (%s, %s)", (obj_id, blob))
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise

    async def get_object(self, key: str) -> bytes:
        async with (await self.get_async_conn_pool()).acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(f"USE {self.schema};")
                await cur.execute(
                    """
                    SELECT d.data
                    FROM object_data d
                    JOIN object_meta m USING(id)
                    WHERE m.path=%s
                """, (key, ))
                row = await cur.fetchone()
                if not row:
                    raise NoSuchKeyError(key=key)
                return pickle.loads(row[0])

    async def delete_object(self, key: str):
        async with (await self.get_async_conn_pool()).acquire() as conn:
            async with conn.cursor() as cur:
                try:
                    await cur.execute(f"USE {self.schema};")
                    await cur.execute(
                        """
                        DELETE m, d
                        FROM object_meta m
                        JOIN object_data d USING(id)
                        WHERE m.path=%s
                    """, (key, ))
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
