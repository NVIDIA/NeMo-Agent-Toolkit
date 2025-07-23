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

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_object_store
from aiq.data_models.object_store import ObjectStoreBaseConfig


class S3ObjectStoreClientConfig(ObjectStoreBaseConfig, name="s3"):
    bucket_name: str
    endpoint_url: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    region: str | None = None


@register_object_store(config_type=S3ObjectStoreClientConfig)
async def s3_object_store_client(config: S3ObjectStoreClientConfig, builder: Builder):

    from aiq.plugins.s3.s3_object_store import S3ObjectStore

    async with S3ObjectStore(config) as store:
        yield store
