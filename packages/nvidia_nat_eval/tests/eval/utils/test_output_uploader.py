# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import subprocess
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest import mock

import pytest

from nat.data_models.dataset_handler import EvalS3Config
from nat.data_models.evaluate_config import EvalCustomScriptConfig
from nat.data_models.evaluate_config import EvalOutputConfig
from nat.plugins.eval.utils.output_uploader import OutputUploader


@pytest.fixture
def s3_config():
    return EvalS3Config(bucket="test-bucket",
                        access_key="fake-access-key",
                        secret_key="fake-secret-key",
                        endpoint_url="https://s3.fake.com")


@pytest.fixture
def output_config(tmp_path, s3_config):
    file = tmp_path / "output.txt"
    file.write_text("some content")
    return EvalOutputConfig(dir=tmp_path, s3=s3_config, remote_dir="my-remote", custom_scripts={})


async def test_upload_directory_success(output_config):
    """Test that the upload_directory uploads the directory to S3 successfully."""
    uploader = OutputUploader(output_config)

    mock_client = mock.Mock()
    with (mock.patch("boto3.client", return_value=mock_client) as mock_boto3_client,
          mock.patch("nat.plugins.eval.utils.output_uploader.asyncio.to_thread", new_callable=mock.AsyncMock) as
          mock_to_thread):
        await uploader.upload_directory()

    expected_key = "my-remote/output.txt"
    local_path = output_config.dir / "output.txt"

    mock_boto3_client.assert_called_once_with(
        "s3",
        endpoint_url=output_config.s3.endpoint_url,
        region_name=None,
        aws_access_key_id=output_config.s3.access_key.get_secret_value(),
        aws_secret_access_key=output_config.s3.secret_key.get_secret_value(),
    )
    mock_to_thread.assert_awaited_once_with(mock_client.upload_file,
                                            str(local_path),
                                            output_config.s3.bucket,
                                            expected_key)
    mock_client.close.assert_called_once_with()


async def test_upload_directory_missing_config(tmp_path):
    """Test that the upload_directory skips uploading if the S3 config is missing."""
    config = EvalOutputConfig(dir=tmp_path, s3=None, remote_dir="", custom_scripts={})
    uploader = OutputUploader(config)

    # Should skip uploading and not raise
    with mock.patch("boto3.client") as mock_client:
        await uploader.upload_directory()

        mock_client.assert_not_called()


async def test_upload_directory_upload_failure(output_config):
    """Test that the upload_directory raises an exception if the upload fails."""
    uploader = OutputUploader(output_config)

    mock_client = mock.Mock()
    mock_client.upload_file.side_effect = Exception("Upload failed")

    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(Exception, match="failed"):
            await uploader.upload_directory()
    mock_client.close.assert_called_once_with()


async def test_upload_directory_waits_for_pending_upload_before_closing(output_config):
    """The S3 client remains open until sibling uploads finish after one fails."""
    pending_path = output_config.dir / "pending.txt"
    pending_path.write_text("pending content")

    uploader = OutputUploader(output_config)
    mock_client = mock.Mock()
    pending_started = asyncio.Event()
    release_pending = asyncio.Event()
    pending_completed = asyncio.Event()

    async def mock_to_thread(_upload_file, local_path, _bucket, _s3_key):
        if local_path == str(pending_path):
            pending_started.set()
            await release_pending.wait()
            await asyncio.sleep(0.01)
            pending_completed.set()
            return

        await pending_started.wait()
        release_pending.set()
        raise RuntimeError("Upload failed")

    def close_client():
        assert pending_completed.is_set()

    mock_client.close.side_effect = close_client
    with (mock.patch("boto3.client", return_value=mock_client),
          mock.patch("nat.plugins.eval.utils.output_uploader.asyncio.to_thread", side_effect=mock_to_thread)):
        with pytest.raises(RuntimeError, match="Upload failed"):
            await uploader.upload_directory()

    mock_client.close.assert_called_once_with()


async def test_upload_directory_missing_boto3_has_install_hint(monkeypatch, output_config):
    """S3 upload should fail with install guidance when optional S3 dependencies are missing."""

    class BlockBoto3:

        def find_spec(self,
                      fullname: str,
                      path: Sequence[str] | None = None,
                      target: ModuleType | None = None) -> ModuleSpec | None:
            if fullname == "boto3" or fullname.startswith("boto3."):
                raise ModuleNotFoundError("No module named 'boto3'")
            return None

    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setattr(sys, "meta_path", [BlockBoto3(), *sys.meta_path])

    with pytest.raises(ModuleNotFoundError, match=r'nvidia-nat-eval\[full\]'):
        await OutputUploader(output_config).upload_directory()


def test_run_custom_scripts_success(tmp_path):
    """Test that the run_custom_scripts runs the custom scripts successfully."""
    script = tmp_path / "dummy_script.py"
    script.write_text("print('Hello nat')")

    config = EvalOutputConfig(dir=tmp_path,
                              s3=None,
                              remote_dir="",
                              custom_scripts={"dummy": EvalCustomScriptConfig(script=script, kwargs={"iam": "ai"})})

    uploader = OutputUploader(config)

    with mock.patch("subprocess.run") as mock_run:
        uploader.run_custom_scripts()
        expected_args = [
            mock.ANY,  # interpreter path
            str(script),
            "--output_dir",
            str(tmp_path),
            "--iam",
            "ai"
        ]
        mock_run.assert_called_once_with(expected_args, check=True, text=True)


def test_run_custom_scripts_missing_script(tmp_path):
    """Test that the run_custom_scripts skips running the custom scripts if the script is missing."""
    missing_script = tmp_path / "not_found.py"

    config = EvalOutputConfig(dir=tmp_path,
                              s3=None,
                              remote_dir="",
                              custom_scripts={"missing": EvalCustomScriptConfig(script=missing_script, kwargs={})})

    uploader = OutputUploader(config)

    with mock.patch("subprocess.run") as mock_run:
        uploader.run_custom_scripts()
        mock_run.assert_not_called()


def test_run_custom_scripts_subprocess_fails(tmp_path):
    script = tmp_path / "fail_script.py"
    script.write_text("raise SystemExit(1)")

    config = EvalOutputConfig(dir=tmp_path,
                              s3=None,
                              remote_dir="",
                              custom_scripts={"fail": EvalCustomScriptConfig(script=script, kwargs={})})

    uploader = OutputUploader(config)

    with mock.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")):
        with pytest.raises(subprocess.CalledProcessError):
            uploader.run_custom_scripts()
