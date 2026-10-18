# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from aiobotocore.stub import AioStubber

from nat.plugins.s3.s3_object_store import S3ObjectStore


@pytest.mark.parametrize(
    "region, default_region, create_params",
    [
        ("us-east-1", "ap-southeast-2", {
            "Bucket": "test-bucket"
        }),
        ("eu-west-1",
         "ap-southeast-2", {
             "Bucket": "test-bucket", "CreateBucketConfiguration": {
                 "LocationConstraint": "eu-west-1"
             }
         }),
        (None,
         "ap-southeast-2", {
             "Bucket": "test-bucket", "CreateBucketConfiguration": {
                 "LocationConstraint": "ap-southeast-2"
             }
         }),
        (None, "us-east-1", {
            "Bucket": "test-bucket"
        }),
        ("aws-global", "ap-southeast-2", {
            "Bucket": "test-bucket"
        }),
        (None, "aws-global", {
            "Bucket": "test-bucket"
        }),
    ])
async def test_create_missing_bucket_in_client_region(monkeypatch, region, default_region, create_params):
    """Create missing buckets using explicit or SDK-default client regions."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", default_region)
    store = S3ObjectStore(bucket_name="test-bucket",
                          endpoint_url=None,
                          access_key="testing",
                          secret_key="testing",
                          region=region)

    async with store.session.client("s3", **store._client_args) as client:
        stubber = AioStubber(client)
        stubber.add_client_error("head_bucket",
                                 service_error_code="404",
                                 http_status_code=404,
                                 expected_params={"Bucket": "test-bucket"})
        stubber.add_response("create_bucket", {}, create_params)
        client_context = AsyncMock()
        client_context.__aenter__.return_value = client

        with stubber, patch.object(store.session, "client", return_value=client_context):
            async with store as opened_store:
                assert opened_store is store
            stubber.assert_no_pending_responses()

        client_context.__aexit__.assert_awaited_once()


async def test_existing_bucket_is_not_created():
    """Leave an existing bucket unchanged and close the client context."""
    store = S3ObjectStore(bucket_name="test-bucket",
                          endpoint_url=None,
                          access_key="testing",
                          secret_key="testing",
                          region="eu-west-1")

    async with store.session.client("s3", **store._client_args) as client:
        stubber = AioStubber(client)
        stubber.add_response("head_bucket", {}, {"Bucket": "test-bucket"})
        client_context = AsyncMock()
        client_context.__aenter__.return_value = client

        with stubber, patch.object(store.session, "client", return_value=client_context):
            async with store:
                pass
            stubber.assert_no_pending_responses()

        client_context.__aexit__.assert_awaited_once()
