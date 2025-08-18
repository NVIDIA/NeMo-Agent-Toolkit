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

import aioboto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError

from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.interfaces import ObjectStore
from nat.object_store.models import ObjectStoreItem
from nat.plugins.s3.object_store import S3ObjectStoreClientConfig

logger = logging.getLogger(__name__)


class S3ObjectStore(ObjectStore):
    """
    S3ObjectStore is an ObjectStore implementation that uses S3 as the underlying storage.
    """

    def __init__(self, config: S3ObjectStoreClientConfig):

        super().__init__()

        self.bucket_name = config.bucket_name
        self.session = aioboto3.Session()
        self._client: BaseClient | None = None
        self._client_context = None

        if not config.access_key:
            raise ValueError("Access key is not set. Please specify it in the environment variable "
                             "'{S3ObjectStoreClientConfig.ACCESS_KEY_ENV}'.")

        if not config.secret_key:
            raise ValueError("Secret key is not set. Please specify it in the environment variable "
                             "'{S3ObjectStoreClientConfig.SECRET_KEY_ENV}'.")

        self._client_args = {
            "aws_access_key_id": config.access_key,
            "aws_secret_access_key": config.secret_key,
            "region_name": config.region,
            "endpoint_url": config.endpoint_url
        }

    async def __aenter__(self):

        if self._client_context is not None:
            raise RuntimeError("Connection already established")

        self._client_context = self.session.client("s3", **self._client_args)
        if self._client_context is None:
            raise RuntimeError("Connection unable to be established")
        self._client = await self._client_context.__aenter__()
        if self._client is None:
            raise RuntimeError("Connection unable to be established")

        # Ensure the bucket exists
        try:
            await self._client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                await self._client.create_bucket(Bucket=self.bucket_name)
                logger.info("Created bucket %s", self.bucket_name)

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):

        if self._client_context is None:
            raise RuntimeError("Connection not established")

        await self._client_context.__aexit__(None, None, None)
        self._client = None
        self._client_context = None

    async def put_object(self, key: str, item: ObjectStoreItem) -> None:

        if self._client is None:
            raise RuntimeError("Connection not established")

        put_args = {
            "Bucket": self.bucket_name,
            "Key": key,
            "Body": item.data,
        }
        if item.content_type:
            put_args["ContentType"] = item.content_type

        if item.metadata:
            put_args["Metadata"] = item.metadata

        try:
            await self._client.put_object(
                **put_args,
                IfNoneMatch='*'  # only succeed if the key does not already exist
            )
        except ClientError as e:
            http_status_code = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", None)
            if http_status_code == 412:
                raise KeyAlreadyExistsError(key=key,
                                            additional_message=f"S3 object {self.bucket_name}/{key} already exists")
            else:
                # Other errors — rethrow or handle accordingly
                raise

    async def upsert_object(self, key: str, item: ObjectStoreItem) -> None:

        if self._client is None:
            raise RuntimeError("Connection not established")

        put_args = {
            "Bucket": self.bucket_name,
            "Key": key,
            "Body": item.data,
        }
        if item.content_type:
            put_args["ContentType"] = item.content_type

        if item.metadata:
            put_args["Metadata"] = item.metadata

        await self._client.put_object(**put_args)

    async def get_object(self, key: str) -> ObjectStoreItem:
        if self._client is None:
            raise RuntimeError("Connection not established")

        try:
            response = await self._client.get_object(Bucket=self.bucket_name, Key=key)
            data = await response["Body"].read()
            return ObjectStoreItem(data=data, content_type=response['ContentType'], metadata=response['Metadata'])
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise NoSuchKeyError(key=key, additional_message=str(e))
            else:
                raise

    async def delete_object(self, key: str) -> None:
        if self._client is None:
            raise RuntimeError("Connection not established")

        try:
            await self._client.get_object(Bucket=self.bucket_name, Key=key)
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise NoSuchKeyError(key=key, additional_message=str(e))
            else:
                raise

        results = await self._client.delete_object(Bucket=self.bucket_name, Key=key)

        if results.get('DeleteMarker', False):
            raise NoSuchKeyError(key=key, additional_message="Object was a delete marker")
