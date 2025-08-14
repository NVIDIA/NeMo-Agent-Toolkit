#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import importlib
import mimetypes
import sys
from pathlib import Path

import click

from nat.builder.workflow_builder import WorkflowBuilder
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem


async def upload_file(object_store: ObjectStore, file_path: Path, key: str):
    """
    Upload a single file to S3/Minio using S3ObjectStore.

    Args:
        object_store: The S3ObjectStore instance to use.
        file_path: The path to the file to upload.
        key: The key to upload the file to.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()

        # Detect content type
        content_type, _ = mimetypes.guess_type(str(file_path))

        # Create ObjectStoreItem
        item = ObjectStoreItem(data=data,
                               content_type=content_type,
                               metadata={
                                   "original_filename": file_path.name, "file_size": str(len(data))
                               })

        # Upload using upsert to allow overwriting
        await object_store.upsert_object(key, item)
        print(f"✅ Uploaded: {file_path.name} -> {key}")

    except Exception as e:
        raise RuntimeError(f"Failed to upload {file_path.name}:\n{e}") from e


async def upload_directory(source_dir: Path, object_store: ObjectStore):
    """
    Upload all files from a directory to S3/Minio using AIQ S3ObjectStore.

    Args:
        source_dir: The local directory to upload.
        object_store: The object store to use.
    """

    try:
        print(f"📁 Processing directory: {source_dir}")
        file_count = 0

        # Process each file recursively
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                key = str(file_path.relative_to(source_dir))
                await upload_file(object_store, file_path, key)
                file_count += 1

        print(f"✅ Directory uploaded successfully! {file_count} files uploaded.")
        return 0

    except Exception as e:
        print(f"❌ Failed to upload directory {source_dir}:\n{e}")
        return 1


@click.command()
@click.option('--store-type',
              type=click.Choice(['s3', 'mysql', 'redis'], case_sensitive=False),
              help='Object store type',
              required=True)
@click.option('--local-dir',
              type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
              help='Directory to upload',
              required=True)
@click.option('--bucket-name', type=str, help='Bucket name', required=True)
@click.option('--host', type=str, help='MySQL or Redis host (optional)')
@click.option('--port', type=int, help='MySQL or Redis port (optional)')
@click.option('--db', type=int, help='Redis db index (optional)')
@click.option('--username', type=str, help='MySQL username (optional)')
@click.option('--password', type=str, help='MySQL password (optional)')
@click.option('--endpoint-url', type=str, help='S3 endpoint URL (optional)')
@click.option('--access-key', type=str, help='S3 access key (optional)')
@click.option('--secret-key', type=str, help='S3 secret key (optional)')
@click.option('--region', type=str, help='S3 region (optional)')
@click.help_option('--help', '-h')
def main(store_type: str,
         local_dir: Path,
         bucket_name: str,
         host: str | None,
         port: int | None,
         db: int | None,
         username: str | None,
         password: str | None,
         endpoint_url: str | None,
         access_key: str | None,
         secret_key: str | None,
         region: str | None):

    async def run():

        all_args = {
            "bucket_name": bucket_name,
            "host": host,
            "port": port,
            "db": db,
            "username": username,
            "password": password,
            "endpoint_url": endpoint_url,
            "access_key": access_key,
            "secret_key": secret_key,
            "region": region
        }

        # Remove all None values
        all_args = {k: v for k, v in all_args.items() if v is not None}

        # Configuration for each store type
        store_configs = {
            's3': {
                'module': 'nat.plugins.s3.object_store', 'config_class': 'S3ObjectStoreClientConfig'
            },
            'mysql': {
                'module': 'nat.plugins.mysql.object_store', 'config_class': 'MySQLObjectStoreClientConfig'
            },
            'redis': {
                'module': 'nat.plugins.redis.object_store', 'config_class': 'RedisObjectStoreClientConfig'
            }
        }

        if store_type not in store_configs:
            raise ValueError(f"Invalid store type: {store_type}. Supported: {list(store_configs.keys())}")

        config = store_configs[store_type]

        try:
            module = importlib.import_module(config['module'])
            config_class = getattr(module, config['config_class'])
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {config['config_class']} from {config['module']}: {e}") from e

        async with WorkflowBuilder() as builder:
            await builder.add_object_store(name=store_type, config=config_class(**all_args))
            store = await builder.get_object_store_client(store_type)
            return await upload_directory(local_dir, store)

    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
