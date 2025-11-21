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
"""Pytest fixtures for S3 integration tests."""

import socket

import pytest


def is_port_open(host: str, port: int) -> bool:
    """Check if a port is open on the given host."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (TimeoutError, ConnectionRefusedError, OSError):
        return False


@pytest.fixture(scope="session")
def minio_server():
    """
    Fixture that checks if MinIO is running on localhost:9000.

    To run S3 integration tests, start MinIO with:

        docker run --rm -p 9000:9000 -p 9001:9001 \\
            -e MINIO_ROOT_USER=minioadmin \\
            -e MINIO_ROOT_PASSWORD=minioadmin \\
            minio/minio server /data --console-address ":9001"

    Then run tests with:
        pytest packages/nvidia_nat_s3/tests/test_s3_object_store.py --run_integration -v
    """
    if not is_port_open("localhost", 9000):
        pytest.skip("MinIO not running on localhost:9000. "
                    "Start MinIO to run S3 integration tests.")

    return {
        "bucket_name": "test-bucket",
        "endpoint_url": "http://localhost:9000",
        "aws_access_key_id": "minioadmin",
        "aws_secret_access_key": "minioadmin",
    }
