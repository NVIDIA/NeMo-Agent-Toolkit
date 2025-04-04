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

import subprocess
from unittest import mock

import pytest

from aiq.data_models.dataset_handler import EvalS3Config
from aiq.data_models.evaluate import EvalCustomScriptConfig
from aiq.data_models.evaluate import EvalOutputConfig
from aiq.eval.utils.output_uploader import OutputUploader


# pylint: disable=redefined-outer-name
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

    mock_client = mock.AsyncMock()
    mock_session = mock.AsyncMock()
    mock_session.__aenter__.return_value = mock_client

    with mock.patch("aioboto3.Session.client", return_value=mock_session):
        await uploader.upload_directory()

    expected_key = "my-remote/output.txt"
    local_path = output_config.dir / "output.txt"

    mock_client.upload_file.assert_called_once_with(str(local_path), output_config.s3.bucket, expected_key)


async def test_upload_directory_missing_config(tmp_path):
    """Test that the upload_directory skips uploading if the S3 config is missing."""
    config = EvalOutputConfig(dir=tmp_path, s3=None, remote_dir="", custom_scripts={})
    uploader = OutputUploader(config)

    # Should skip uploading and not raise
    with mock.patch("aioboto3.Session.client") as mock_client:
        mock_client.return_value = mock.AsyncMock()
        await uploader.upload_directory()

        mock_client.assert_not_called()


async def test_upload_directory_upload_failure(output_config):
    """Test that the upload_directory raises an exception if the upload fails."""
    uploader = OutputUploader(output_config)

    mock_client = mock.AsyncMock()
    mock_client.upload_file.side_effect = Exception("Upload failed")

    mock_session = mock.AsyncMock()
    mock_session.__aenter__.return_value = mock_client

    with mock.patch("aioboto3.Session.client", return_value=mock_session):
        with pytest.raises(Exception, match="failed"):
            await uploader.upload_directory()


def test_run_custom_scripts_success(tmp_path):
    """Test that the run_custom_scripts runs the custom scripts successfully."""
    script = tmp_path / "dummy_script.py"
    script.write_text("print('Hello aiq')")

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
