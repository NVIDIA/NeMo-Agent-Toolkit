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

import sys

import pytest

from nat.data_models.dataset_handler import EvalDatasetJsonConfig
from nat.data_models.dataset_handler import EvalS3Config
from nat.plugins.eval.dataset_handler.dataset_downloader import DatasetDownloader


class _BlockModules:

    def __init__(self, module_roots: set[str]):
        self._module_roots = module_roots

    def find_spec(self, fullname, path=None, target=None):  # noqa: ANN001
        if any(fullname == root or fullname.startswith(f"{root}.") for root in self._module_roots):
            raise ModuleNotFoundError(f"No module named '{fullname}'")


def test_signed_url_download_missing_requests_has_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "requests", None)
    monkeypatch.setattr(sys, "meta_path", [_BlockModules({"requests"}), *sys.meta_path])

    downloader = DatasetDownloader(EvalDatasetJsonConfig())

    with pytest.raises(ModuleNotFoundError, match=r"nvidia-nat-eval\[full\]"):
        downloader.download_with_signed_url("https://example.com/dataset.json", str(tmp_path / "dataset.json"))


def test_s3_download_missing_boto3_has_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setitem(sys.modules, "botocore", None)
    monkeypatch.setattr(sys, "meta_path", [_BlockModules({"boto3", "botocore"}), *sys.meta_path])

    config = EvalDatasetJsonConfig(
        file_path=tmp_path / "dataset.json",
        remote_file_path="dataset.json",
        s3=EvalS3Config(bucket="bucket", access_key="access-key", secret_key="secret-key"),
    )
    downloader = DatasetDownloader(config)

    with pytest.raises(ModuleNotFoundError, match=r"nvidia-nat-eval\[full\]"):
        downloader.download_with_boto3("dataset.json", str(tmp_path / "dataset.json"))
