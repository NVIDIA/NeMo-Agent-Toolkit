#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import pickle
from pathlib import Path

import aiomysql
import click

from aiq.object_store.models import ObjectStoreItem


async def create_database_and_tables(connection, db_name: str):
    """
    Create the database and required tables.

    Args:
        connection: The database connection.
        db_name: The name of the database.
    """
    async with connection.cursor() as cursor:
        # Drop database if exists
        await cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")

        # Create database
        await cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}` DEFAULT CHARACTER SET utf8mb4")
        await cursor.execute(f"USE `{db_name}`")

        # Create object_meta table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS object_meta (
                id INT AUTO_INCREMENT PRIMARY KEY,
                path VARCHAR(768) NOT NULL UNIQUE,
                size BIGINT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB
        """)

        # Create object_data table
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS object_data (
                id INT PRIMARY KEY,
                data LONGBLOB NOT NULL,
                FOREIGN KEY (id) REFERENCES object_meta(id) ON DELETE CASCADE
            ) ENGINE=InnoDB ROW_FORMAT=DYNAMIC
        """)

        await connection.commit()


async def store_file(connection, db_name: str, file_path: Path, relative_path: str):
    """
    Store a single file in the database.

    Args:
        connection: The database connection.
        db_name: The name of the database.
        file_path: The path to the file.
        relative_path: The relative path to the file.
    """
    file_size = file_path.stat().st_size

    # Serialize the file
    with open(file_path, "rb") as f:
        data = f.read()

    item = ObjectStoreItem(data=data)
    serialized_data = pickle.dumps(item)

    async with connection.cursor() as cursor:
        await cursor.execute(f"USE `{db_name}`")

        # Start transaction
        await cursor.execute("START TRANSACTION")

        try:
            # Insert or update metadata
            await cursor.execute(
                """
                INSERT INTO object_meta (path, size)
                VALUES (%s, %s) AS new_values
                ON DUPLICATE KEY UPDATE size=new_values.size, created_at=CURRENT_TIMESTAMP
            """, (relative_path, file_size))

            # Get the object ID
            await cursor.execute("SELECT id FROM object_meta WHERE path=%s", (relative_path, ))
            result = await cursor.fetchone()
            obj_id = result[0]

            # Store the serialized data
            await cursor.execute(
                """
                REPLACE INTO object_data (id, data)
                VALUES (%s, %s)
            """, (obj_id, serialized_data))

            # Commit transaction
            await cursor.execute("COMMIT")
            print(f"✅ Stored '{relative_path}', {file_size} bytes.")

        except Exception as e:
            await cursor.execute("ROLLBACK")
            print(f"❌ Error storing '{relative_path}': {e}")
            raise


async def process_directory(db_host: str,
                            db_port: int,
                            db_user: str,
                            db_password: str | None,
                            source_dir: Path,
                            bucket_name: str):
    """
    Process all files in the directory and upload to MySQL.

    Args:
        db_user: The database username.
        db_pass: The database password.
        source_dir: The source directory.
        bucket_name: The bucket name.
    """

    if not db_password:
        print("""❌ Error: Database password is required!
Please provide it via --db-password or set MYSQL_ROOT_PASSWORD environment variable.""")
        return 1

    db_name = f"bucket_{bucket_name}"

    # Connect to MySQL
    connection = await aiomysql.connect(host=db_host,
                                        port=db_port,
                                        user=db_user,
                                        password=db_password,
                                        autocommit=False)

    try:
        print(f"🪣 Target database: {db_name}")
        # Create database and tables
        await create_database_and_tables(connection, db_name)

        print(f"📁 Processing directory: {source_dir}")
        # Process each file
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                relative_path = str(file_path.relative_to(source_dir))
                await store_file(connection, db_name, file_path, relative_path)

    finally:
        await connection.ensure_closed()


@click.command()
@click.argument('local-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--db-host', type=str, default="localhost", help='Database host. Default: localhost')
@click.option('--db-port', type=int, default=3306, help='Database port. Default: 3306')
@click.option('--db-user', type=str, default="root", help='Database username. Default: root')
@click.option('--db-password',
              type=str,
              required=False,
              help='Database password. Default: content of MYSQL_ROOT_PASSWORD env var')
@click.argument('bucket_name', type=str)
@click.help_option('--help', '-h')
def main(local_directory: Path, bucket_name: str, db_host: str, db_port: int, db_user: str, db_password: str | None):
    return asyncio.run(
        process_directory(db_host=db_host,
                          db_port=db_port,
                          db_user=db_user,
                          db_password=db_password or os.environ.get("MYSQL_ROOT_PASSWORD"),
                          source_dir=local_directory,
                          bucket_name=bucket_name))


if __name__ == "__main__":
    exit(main())
