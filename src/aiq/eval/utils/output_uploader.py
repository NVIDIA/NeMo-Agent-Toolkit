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
import os
import subprocess
from pathlib import Path

import boto3
from botocore.exceptions import NoCredentialsError

from aiq.data_models.evaluate import EvalOutputConfig

logger = logging.getLogger(__name__)


class OutputUploader:
    """
    Run custom scripts and upload evaluation outputs to S3.
    """

    def __init__(self, output_config: EvalOutputConfig):
        self.output_config = output_config
        self._s3_client = None

    @property
    def s3_config(self):
        return self.output_config.s3

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

    def upload_directory(self):
        """
        Upload the contents of the local output directory to the remote S3 bucket.
        Preserves relative file structure.
        """
        if not self.output_config.s3:
            logger.info("No S3 config provided; skipping upload.")
            return

        local_dir = self.output_config.dir
        bucket = self.s3_config.bucket
        remote_prefix = self.output_config.remote_dir or ""

        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = Path(root) / file
                relative_path = local_path.relative_to(local_dir)
                s3_path = Path(remote_prefix) / relative_path
                s3_key = str(s3_path).replace("\\", "/")  # Normalize for S3

                try:
                    self.s3_client.upload_file(str(local_path), bucket, s3_key)
                    logger.info("Uploaded %s to s3://%s/%s", local_path, bucket, s3_key)
                except Exception as e:
                    logger.error("Failed to upload %s to s3://%s/%s: %s", local_path, bucket, s3_key, e)
                    raise

    def run_custom_scripts(self):
        """
        Run custom scripts defined in the EvalOutputConfig.
        Each script is run with its kwargs passed as command-line arguments (if any).
        """
        for script_config in self.output_config.custom_script:
            script_path = script_config.script
            if not script_path.exists():
                logger.error("Custom script %s does not exist.", script_path)
                continue

            args = [str(script_path)]

            if script_config.kwargs:
                for key, value in script_config.kwargs.items():
                    args.append(f"--{key}")
                    args.append(str(value))

            try:
                logger.info("Running custom script: %s %s", script_path, " ".join(args[1:]))
                subprocess.run(args, check=True)
                logger.info("Custom script %s completed successfully.", script_path)
            except subprocess.CalledProcessError as e:
                logger.error("Custom script %s failed: %s", script_path, e)
                raise
