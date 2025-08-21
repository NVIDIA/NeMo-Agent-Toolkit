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

import asyncio
import importlib
import logging
import mimetypes
from pathlib import Path

import click

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)

# Supported object store types
SUPPORTED_STORE_TYPES = ['s3', 'mysql', 'redis']


def object_store_common_options(func):
    """Decorator to add common object store options to commands."""
    func = click.option('--store-type',
                        type=click.Choice(SUPPORTED_STORE_TYPES, case_sensitive=False),
                        help='Object store type',
                        required=True)(func)
    func = click.option('--bucket-name', type=str, help='Bucket name', required=True)(func)
    func = click.option('--host', type=str, help='MySQL or Redis host (optional)')(func)
    func = click.option('--port', type=int, help='MySQL or Redis port (optional)')(func)
    func = click.option('--db', type=int, help='Redis db index (optional)')(func)
    func = click.option('--username', type=str, help='MySQL username (optional)')(func)
    func = click.option('--password', type=str, help='MySQL password (optional)')(func)
    func = click.option('--endpoint-url', type=str, help='S3 endpoint URL (optional)')(func)
    func = click.option('--access-key', type=str, help='S3 access key (optional)')(func)
    func = click.option('--secret-key', type=str, help='S3 secret key (optional)')(func)
    func = click.option('--region', type=str, help='S3 region (optional)')(func)
    func = click.help_option('--help', '-h')(func)
    return func


def get_object_store_config(**kwargs) -> ObjectStoreBaseConfig:
    """Process common object store arguments and return config dict and class."""
    all_args = {k: v for k, v in kwargs.items() if v is not None}
    store_type = all_args.pop('store_type')
    config_class = get_config_class(store_type)
    return config_class(**all_args)


def get_config_class(store_type: str) -> type[ObjectStoreBaseConfig]:
    """Get the config class for a given store type."""
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
        raise ValueError(f"Invalid store type: {store_type}. "
                         f"Supported: {SUPPORTED_STORE_TYPES}")

    config = store_configs[store_type]

    try:
        module = importlib.import_module(config['module'])
        return getattr(module, config['config_class'])
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import {config['config_class']} "
                          f"from {config['module']}: {e}") from e


async def upload_file(object_store: ObjectStore, file_path: Path, key: str):
    """
    Upload a single file to object store.

    Args:
        object_store: The object store instance to use.
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
                                   "original_filename": file_path.name,
                                   "file_size": str(len(data)),
                                   "file_extension": file_path.suffix,
                                   "upload_timestamp": str(int(asyncio.get_event_loop().time()))
                               })

        # Upload using upsert to allow overwriting
        await object_store.upsert_object(key, item)
        click.echo(f"✅ Uploaded: {file_path.name} -> {key}")

    except Exception as e:
        raise click.ClickException(f"Failed to upload {file_path.name}:\n{e}") from e


def object_store_command_decorator(async_func):
    """
    Decorator that handles the common object store command pattern.

    The decorated function should take (store: ObjectStore, kwargs) as parameters
    and return an exit code (0 for success).
    """

    def wrapper(store_type: str, **kwargs):
        config = get_object_store_config(store_type=store_type, **kwargs)

        async def work():
            async with WorkflowBuilder() as builder:
                await builder.add_object_store(name=store_type, config=config)
                store = await builder.get_object_store_client(store_type)
                return await async_func(store, **kwargs)

        try:
            exit_code = asyncio.run(work())
            if exit_code != 0:
                raise click.ClickException(f"Command failed with exit code {exit_code}")
        except Exception as e:
            raise click.ClickException(f"Command failed: {e}")

    return wrapper


@click.command(name="upload", help="Upload a directory to an object store.")
@click.argument('local_dir',
                type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
                required=True)
@object_store_common_options
@object_store_command_decorator
async def upload_command(store: ObjectStore, local_dir: Path, **_kwargs):
    """
    Upload a directory to an object store.

    Args:
        local_dir: The local directory to upload.
        store: The object store to use.
        _kwargs: Additional keyword arguments.
    """
    try:
        click.echo(f"📁 Processing directory: {local_dir}")
        file_count = 0

        # Process each file recursively
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                key = str(file_path.relative_to(local_dir))
                await upload_file(store, file_path, key)
                file_count += 1

        click.echo(f"✅ Directory uploaded successfully! {file_count} files uploaded.")
        return 0

    except Exception as e:
        raise click.ClickException(f"❌ Failed to upload directory {local_dir}:\n  {e}") from e


@click.command(name="delete", help="Delete files from an object store.")
@click.argument('keys', type=str, required=True, nargs=-1)
@object_store_common_options
@object_store_command_decorator
async def delete_command(store: ObjectStore, keys: list[str], **_kwargs):
    """
    Delete files from an object store.

    Args:
        store: The object store to use.
        keys: The keys to delete.
        _kwargs: Additional keyword arguments.
    """
    deleted_count = 0
    for key in keys:
        try:
            await store.delete_object(key)
            click.echo(f"✅ Deleted: {key}")
            deleted_count += 1
        except Exception as e:
            raise click.ClickException(f"❌ Failed to delete {key}:\n  {e}") from e

    click.echo(f"✅ Deletion completed! {deleted_count} keys deleted.")
    return 0


@click.group(name="object-store", invoke_without_command=False, help="Manage object store operations.")
def object_store_command(**_kwargs):
    """Manage object store operations including uploading files and directories."""
    pass


object_store_command.add_command(upload_command, name="upload")
object_store_command.add_command(delete_command, name="delete")
