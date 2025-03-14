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
import logging
from pathlib import Path

import boto3
import requests
from botocore.exceptions import NoCredentialsError

from aiq.data_models.dataset_handler import EvalDatasetConfig

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Download remote datasets using signed URLs or S3 credentials.

    One DatasetDownloader object is needed for each dataset to be downloaded.
    """

    def __init__(self, dataset_config: EvalDatasetConfig):
        self.dataset_config = dataset_config
        self._s3_client = None

    @property
    def s3_config(self):
        return self.dataset_config.s3

    @property
    def s3_client(self):
        """Lazy init the S3 client."""
        if not self._s3_client:
            try:
                self._s3_client = boto3.client("s3",
                                               endpoint_url=self.s3_config.endpoint_url,
                                               aws_access_key_id=self.s3_config.access_key,
                                               aws_secret_access_key=self.s3_config.secret_key)
            except NoCredentialsError as e:
                logger.error("AWS credentials not available: %s", e)
                raise
            except Exception as e:
                logger.error("Failed to initialize S3 client: %s", e)
                raise
        return self._s3_client

    @staticmethod
    def ensure_directory_exists(file_path: str):
        """Ensure the directory for the file exists."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    def download_with_signed_url(self, remote_file_path: str, local_file_path: str, timeout: int = 300):
        """Download a file using a signed URL."""
        try:
            response = requests.get(remote_file_path, stream=True, timeout=timeout)
            response.raise_for_status()
            with open(local_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            logger.info("File downloaded successfully to %s using signed URL.", local_file_path)
        except requests.exceptions.RequestException as e:
            logger.error("Error downloading file using signed URL: %s", e)
            raise

    def download_with_boto3(self, remote_file_path: str, local_file_path: str):
        """Download a file using boto3 and credentials."""
        try:
            self.s3_client.download_file(self.dataset_config.s3.bucket, remote_file_path, local_file_path)
            logger.info("File downloaded successfully to %s using S3 client.", local_file_path)
        except Exception as e:
            logger.error("Error downloading file from S3: %s", e)
            raise

    @staticmethod
    def is_file_path_url(file_path: str) -> bool:
        """Check if the file path is a URL."""
        return file_path.startswith("http")

    def download_file(self, remote_file_path: str, local_file_path: str):
        """Download a file using the appropriate method."""
        self.ensure_directory_exists(local_file_path)
        if self.is_file_path_url(remote_file_path):
            logger.info("Using signed URL to download the file %s...", remote_file_path)
            self.download_with_signed_url(remote_file_path, local_file_path, timeout=120)
        else:
            logger.info("Using S3 credentials to download the file %s...", remote_file_path)
            self.download_with_boto3(remote_file_path, local_file_path)

    def download_dataset(self):
        """Download datasets defined in the evaluation configuration."""
        if self.dataset_config.remote_file_path:
            logger.info("Downloading remote dataset %s")
            self.download_file(remote_file_path=self.dataset_config.remote_file_path,
                               local_file_path=self.dataset_config.file_path)
