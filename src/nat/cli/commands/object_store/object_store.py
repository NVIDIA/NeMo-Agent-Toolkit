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
import time
from pathlib import Path

import click

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.object_store import ObjectStoreBaseConfig
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)

STORE_CONFIGS = {
    "s3": {
        "module": "nat.plugins.s3.object_store", "config_class": "S3ObjectStoreClientConfig"
    },
    "mysql": {
        "module": "nat.plugins.mysql.object_store", "config_class": "MySQLObjectStoreClientConfig"
    },
    "redis": {
        "module": "nat.plugins.redis.object_store", "config_class": "RedisObjectStoreClientConfig"
    }
}


def get_object_store_config(**kwargs) -> ObjectStoreBaseConfig:
    """Process common object store arguments and return the config class"""
    store_type = kwargs.pop("store_type")
    config = STORE_CONFIGS[store_type]
    module = importlib.import_module(config["module"])
    config_class = getattr(module, config["config_class"])
    return config_class(**kwargs)


async def upload_file(object_store: ObjectStore, file_path: Path, key: str):
    """
    Upload a single file to object store.

    Args:
        object_store: The object store instance to use.
        file_path: The path to the file to upload.
        key: The key to upload the file to.
    """
    try:
        data = await asyncio.to_thread(file_path.read_bytes)

        item = ObjectStoreItem(data=data,
                               content_type=mimetypes.guess_type(str(file_path))[0],
                               metadata={
                                   "original_filename": file_path.name,
                                   "file_size": str(len(data)),
                                   "file_extension": file_path.suffix,
                                   "upload_timestamp": str(int(time.time()))
                               })

        # Upload using upsert to allow overwriting
        await object_store.upsert_object(key, item)
        click.echo(f"✅ Uploaded: {file_path.name} -> {key}")

    except Exception as e:
        raise RuntimeError(f"Failed to upload {file_path.name}:\n{e}") from e


def object_store_command_decorator(async_func):
    """
    Decorator that handles the common object store command pattern.

    The decorated function should take (store: ObjectStore, kwargs) as parameters
    and return an exit code (0 for success).
    """

    @click.pass_context
    def wrapper(ctx: click.Context, **kwargs):
        config = ctx.obj["store_config"]

        async def work():
            async with WorkflowBuilder() as builder:
                await builder.add_object_store(name="store", config=config)
                store = await builder.get_object_store_client("store")
                return await async_func(store, **kwargs)

        try:
            exit_code = asyncio.run(work())
        except Exception as e:
            raise click.ClickException(f"Command failed: {e}") from e
        if exit_code != 0:
            raise click.ClickException(f"Command failed with exit code {exit_code}")
        return exit_code

    return wrapper


@click.command(name="upload", help="Upload a directory to an object store.")
@click.argument("local_dir",
                type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
                required=True)
@click.help_option("--help", "-h")
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
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                key = file_path.relative_to(local_dir).as_posix()
                await upload_file(store, file_path, key)
                file_count += 1

        click.echo(f"✅ Directory uploaded successfully! {file_count} files uploaded.")
        return 0

    except Exception as e:
        raise click.ClickException(f"❌ Failed to upload directory {local_dir}:\n  {e}") from e


@click.command(name="delete", help="Delete files from an object store.")
@click.argument("keys", type=str, required=True, nargs=-1)
@click.help_option("--help", "-h")
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
    failed_count = 0
    for key in keys:
        try:
            await store.delete_object(key)
            click.echo(f"✅ Deleted: {key}")
            deleted_count += 1
        except Exception as e:
            click.echo(f"❌ Failed to delete {key}: {e}")
            failed_count += 1

    click.echo(f"✅ Deletion completed! {deleted_count} keys deleted. {failed_count} keys failed to delete.")
    return 0 if failed_count == 0 else 1


@click.group(name="object-store", invoke_without_command=False, help="Manage object store operations.")
def object_store_command(**_kwargs):
    """Manage object store operations including uploading files and directories."""
    pass


def register_object_store_commands():

    @click.group(name="s3", invoke_without_command=False, help="S3 object store operations.")
    @click.argument("bucket_name", type=str, required=True)
    @click.option("--endpoint-url", type=str, help="S3 endpoint URL")
    @click.option("--access-key", type=str, help="S3 access key")
    @click.option("--secret-key", type=str, help="S3 secret key")
    @click.option("--region", type=str, help="S3 region")
    @click.pass_context
    def s3(ctx: click.Context, **kwargs):
        ctx.ensure_object(dict)
        ctx.obj["store_config"] = get_object_store_config(store_type="s3", **kwargs)

    @click.group(name="mysql", invoke_without_command=False, help="MySQL object store operations.")
    @click.argument("bucket_name", type=str, required=True)
    @click.option("--host", type=str, help="MySQL host")
    @click.option("--port", type=int, help="MySQL port")
    @click.option("--db", type=str, help="MySQL database name")
    @click.option("--username", type=str, help="MySQL username")
    @click.option("--password", type=str, help="MySQL password")
    @click.pass_context
    def mysql(ctx: click.Context, **kwargs):
        ctx.ensure_object(dict)
        ctx.obj["store_config"] = get_object_store_config(store_type="mysql", **kwargs)

    @click.group(name="redis", invoke_without_command=False, help="Redis object store operations.")
    @click.argument("bucket_name", type=str, required=True)
    @click.option("--host", type=str, help="Redis host")
    @click.option("--port", type=int, help="Redis port")
    @click.option("--db", type=int, help="Redis db")
    @click.pass_context
    def redis(ctx: click.Context, **kwargs):
        ctx.ensure_object(dict)
        ctx.obj["store_config"] = get_object_store_config(store_type="redis", **kwargs)

    commands = {"s3": s3, "mysql": mysql, "redis": redis}

    for store_type, config in STORE_CONFIGS.items():
        try:
            importlib.import_module(config["module"])
            command = commands[store_type]
            object_store_command.add_command(command, name=store_type)
            command.add_command(upload_command, name="upload")
            command.add_command(delete_command, name="delete")
        except ImportError:
            pass


register_object_store_commands()
