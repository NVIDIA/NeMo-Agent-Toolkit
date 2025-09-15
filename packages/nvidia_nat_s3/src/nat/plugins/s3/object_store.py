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


class S3ObjectStoreClientConfig(ObjectStoreBaseConfig, name="s3"):
    """
    Object store that stores objects in an S3 bucket.
    """

    ACCESS_KEY_ENV: ClassVar[str] = "NAT_S3_OBJECT_STORE_ACCESS_KEY"
    SECRET_KEY_ENV: ClassVar[str] = "NAT_S3_OBJECT_STORE_SECRET_KEY"

    bucket_name: str = Field(..., description="The name of the bucket to use for the object store")
    endpoint_url: str | None = Field(default=None, description="The URL of the S3 server to connect to")
    access_key: str | None = Field(default=os.environ.get(ACCESS_KEY_ENV),
                                   description=f"Access key. If omitted, reads from {ACCESS_KEY_ENV}")
    secret_key: str | None = Field(default=os.environ.get(SECRET_KEY_ENV),
                                   description=f"Secret key. If omitted, reads from {SECRET_KEY_ENV}")
    region: str | None = Field(default=None, description="Region to access (or none if unspecified)")


@register_object_store(config_type=S3ObjectStoreClientConfig)
async def s3_object_store_client(config: S3ObjectStoreClientConfig, _builder: Builder):

    from .s3_object_store import S3ObjectStore

    async with S3ObjectStore(**config.model_dump(exclude={"type"}, exclude_none=True)) as store:
        yield store
