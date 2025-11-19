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
"""Integration tests for video upload, list, and delete routes using InMemoryObjectStore"""

import io

import pytest

from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from nat.object_store.in_memory_object_store import InMemoryObjectStoreConfig
from nat.test.functions import EchoFunctionConfig
from nat.test.utils import build_nat_client


@pytest.fixture
def object_store_name():
    return "test_video_store"


@pytest.fixture
async def client(object_store_name):
    """Create an async test client for the FastAPI app with InMemoryObjectStore"""
    config = Config(
        general=GeneralConfig(front_end=FastApiFrontEndConfig(object_store=object_store_name)),
        object_stores={object_store_name: InMemoryObjectStoreConfig()},
        workflow=EchoFunctionConfig(),  # Dummy workflow for tests
    )

    async with build_nat_client(config) as test_client:
        yield test_client


@pytest.mark.asyncio
class TestVideoUploadRoutes:
    """Test video upload, list, and delete functionality with InMemoryObjectStore"""

    async def test_upload_video_success(self, client):
        """Test successful video upload"""

        video_data = b"fake_video_data_mp4"

        response = await client.post("/videos",
                                     files={"file": ("test_video.mp4", io.BytesIO(video_data), "video/mp4")},
                                     data={"metadata": "test metadata"})

        assert response.status_code == 200
        result = response.json()

        assert "video_key" in result
        assert result["video_key"].startswith("videos/")
        assert result["filename"] == "test_video.mp4"
        assert result["content_type"] == "video/mp4"
        assert result["size"] == len(video_data)
        assert "uuid" in result

    async def test_upload_non_video_file_rejected(self, client):
        """Test that non-video files are rejected"""

        text_data = b"not a video"

        response = await client.post("/videos", files={"file": ("test.txt", io.BytesIO(text_data), "text/plain")})

        assert response.status_code == 400
        assert "video" in response.json()["detail"].lower()

    async def test_list_videos_empty(self, client):
        """Test listing videos when none exist"""

        response = await client.get("/videos")

        assert response.status_code == 200
        result = response.json()
        assert "videos" in result
        assert isinstance(result["videos"], list)

    async def test_list_videos_after_upload(self, client):
        """Test listing videos after uploading some"""

        video1_data = b"fake_video_1"
        response1 = await client.post("/videos", files={"file": ("video1.mp4", io.BytesIO(video1_data), "video/mp4")})
        assert response1.status_code == 200
        video1_key = response1.json()["video_key"]

        video2_data = b"fake_video_2"
        response2 = await client.post("/videos", files={"file": ("video2.avi", io.BytesIO(video2_data), "video/avi")})
        assert response2.status_code == 200
        video2_key = response2.json()["video_key"]

        # List videos
        response = await client.get("/videos")
        assert response.status_code == 200
        result = response.json()

        # Find our uploaded videos
        video_keys = {v["video_key"] for v in result["videos"]}
        assert video1_key in video_keys
        assert video2_key in video_keys

        # Verify video metadata
        for video in result["videos"]:
            if video["video_key"] == video1_key:
                assert video["filename"] == "video1.mp4"
                assert video["size"] == len(video1_data)
            elif video["video_key"] == video2_key:
                assert video["filename"] == "video2.avi"
                assert video["size"] == len(video2_data)

    async def test_delete_video_success(self, client):
        """Test successful video deletion"""

        video_data = b"video_to_delete"
        response = await client.post("/videos", files={"file": ("delete_me.mp4", io.BytesIO(video_data), "video/mp4")})
        assert response.status_code == 200
        video_key = response.json()["video_key"]

        delete_response = await client.delete(f"/videos/{video_key}")
        assert delete_response.status_code == 200
        result = delete_response.json()
        assert "deleted successfully" in result["message"]
        assert result["video_key"] == video_key

        list_response = await client.get("/videos")
        video_keys = {v["video_key"] for v in list_response.json()["videos"]}
        assert video_key not in video_keys

    async def test_delete_video_idempotent(self, client):
        """Test that deleting a non-existent video is idempotent (doesn't error)"""

        # Try to delete a video that doesn't exist
        fake_key = "videos/00000000-0000-0000-0000-000000000000/nonexistent.mp4"
        response = await client.delete(f"/videos/{fake_key}")

        assert response.status_code == 200
        result = response.json()
        assert "deleted successfully" in result["message"].lower()

    async def test_delete_invalid_key_rejected(self, client):
        """Test that invalid video keys are rejected"""

        # Try to delete with a key that doesn't start with "videos/"
        response = await client.delete("/videos/invalid/path/video.mp4")

        assert response.status_code == 400
        assert "invalid" in response.json()["detail"].lower()

    async def test_full_workflow(self, client):
        """Test complete upload → list → delete → list workflow"""

        video_data = b"workflow_test_video"
        upload_response = await client.post("/videos",
                                            files={"file": ("workflow.mp4", io.BytesIO(video_data), "video/mp4")},
                                            data={"metadata": "workflow test"})
        assert upload_response.status_code == 200
        video_key = upload_response.json()["video_key"]

        list_response_1 = await client.get("/videos")
        assert list_response_1.status_code == 200
        video_keys_1 = {v["video_key"] for v in list_response_1.json()["videos"]}
        assert video_key in video_keys_1

        delete_response = await client.delete(f"/videos/{video_key}")
        assert delete_response.status_code == 200

        list_response_2 = await client.get("/videos")
        assert list_response_2.status_code == 200
        video_keys_2 = {v["video_key"] for v in list_response_2.json()["videos"]}
        assert video_key not in video_keys_2
