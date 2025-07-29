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
import mimetypes
import os
import sys
from pathlib import Path

import click

from aiq.object_store.models import ObjectStoreItem
from aiq.plugins.s3.object_store import S3ObjectStoreClientConfig
from aiq.plugins.s3.s3_object_store import S3ObjectStore


async def upload_file(object_store: S3ObjectStore, file_path: Path, key: str):
    """
    Upload a single file to S3/Minio using AIQ S3ObjectStore.

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
        print(f"❌ Failed to upload {file_path.name}: {e}")
        raise


async def upload_directory(source_dir: Path,
                           endpoint_url: str,
                           bucket_name: str,
                           bucket_prefix: str = "",
                           access_key: str | None = None,
                           secret_key: str | None = None,
                           region: str | None = None):
    """
    Upload all files from a directory to S3/Minio using AIQ S3ObjectStore.

    Args:
        source_dir: The local directory to upload.
        endpoint_url: The URL of the S3/MinIO endpoint.
        bucket_name: The name of the bucket to upload to.
        bucket_prefix: The prefix to upload the files to.
        access_key: S3 access key (optional, can use env vars).
        secret_key: S3 secret key (optional, can use env vars).
        region: S3 region (optional).
    """

    # Create S3ObjectStore configuration
    config = S3ObjectStoreClientConfig(bucket_name=bucket_name,
                                       endpoint_url=endpoint_url,
                                       access_key=access_key,
                                       secret_key=secret_key,
                                       region=region)

    try:
        async with S3ObjectStore(config) as object_store:
            print(f"📁 Processing directory: {source_dir}")
            print(f"🪣 Target bucket: {bucket_name}")
            if bucket_prefix:
                print(f"📂 Bucket prefix: {bucket_prefix}")

            file_count = 0

            # Process each file recursively
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(source_dir))
                    # Construct the key with bucket prefix
                    key = f"{bucket_prefix.rstrip('/')}/{relative_path}" if bucket_prefix else relative_path

                    await upload_file(object_store, file_path, key)
                    file_count += 1

            print(f"✅ Upload completed successfully! {file_count} files uploaded.")
            return 0

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1


DEFAULT_ENDPOINT_URL: str = "http://localhost:9000"


@click.command()
@click.argument('local_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.argument('bucket_name', type=str)
@click.option('--bucket-prefix',
              required=False,
              type=str,
              default="",
              help="Optional prefix path within the bucket. Default: \"\"")
@click.option('--endpoint-url',
              type=str,
              default=DEFAULT_ENDPOINT_URL,
              help=f"S3/MinIO endpoint URL. Default: {DEFAULT_ENDPOINT_URL}")
@click.option('--access-key', type=str, help='S3 access key (or set AIQ_S3_OBJECT_STORE_ACCESS_KEY)')
@click.option('--secret-key', type=str, help='S3 secret key (or set AIQ_OBJECT_STORE_SECRET_KEY)')
@click.option('--region', type=str, help='S3 region (optional)')
@click.help_option('--help', '-h')
def main(local_directory: Path,
         bucket_name: str,
         bucket_prefix: str,
         endpoint_url: str,
         access_key: str | None = None,
         secret_key: str | None = None,
         region: str | None = None):
    if not access_key:
        access_key = os.environ.get("AIQ_S3_OBJECT_STORE_ACCESS_KEY") or "minioadmin"
    if not secret_key:
        secret_key = os.environ.get("AIQ_OBJECT_STORE_SECRET_KEY") or "minioadmin"

    return asyncio.run(
        upload_directory(
            source_dir=local_directory,
            endpoint_url=endpoint_url,
            bucket_name=bucket_name,
            bucket_prefix=bucket_prefix,
            access_key=access_key,
            secret_key=secret_key,
            region=region,
        ))


if __name__ == "__main__":
    sys.exit(main())
