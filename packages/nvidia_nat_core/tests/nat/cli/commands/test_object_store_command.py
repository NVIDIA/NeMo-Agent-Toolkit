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

import pytest

from nat.cli.commands.object_store.object_store import STORE_CONFIGS
from nat.cli.commands.object_store.object_store import get_object_store_config
from nat.data_models.object_store import ObjectStoreBaseConfig


class SampleObjectStoreClientConfig(ObjectStoreBaseConfig, name="sample_object_store"):
    """Mirrors the shape of the first-party store configs: required argument plus defaulted options."""

    bucket_name: str
    host: str = "localhost"
    port: int = 3306
    username: str | None = "config-default-user"


@pytest.fixture(name="sample_store_type")
def sample_store_type_fixture(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setitem(STORE_CONFIGS, "sample", {"module": __name__, "config_class": "SampleObjectStoreClientConfig"})
    return "sample"


def test_get_object_store_config_keeps_supplied_options(sample_store_type: str):
    config = get_object_store_config(store_type=sample_store_type,
                                     bucket_name="bucket",
                                     host="db.example.com",
                                     port=3307,
                                     username="alice")

    assert isinstance(config, SampleObjectStoreClientConfig)
    assert config.host == "db.example.com"
    assert config.port == 3307
    assert config.username == "alice"


def test_get_object_store_config_falls_back_to_config_defaults(sample_store_type: str):
    # Click passes `None` for every option the user did not spell out on the command line.
    config = get_object_store_config(store_type=sample_store_type,
                                     bucket_name="bucket",
                                     host=None,
                                     port=None,
                                     username=None)

    assert config.host == "localhost"
    assert config.port == 3306
    assert config.username == "config-default-user"
