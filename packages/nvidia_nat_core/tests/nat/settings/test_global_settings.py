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

import logging

import pytest

from nat.data_models.common import get_secret_value
from nat.settings.global_settings import GlobalSettings
from nat.settings.global_settings import Settings
from nat.utils.settings.global_settings import remove_channel


@pytest.fixture(name="config_dir")
def config_dir_fixture(tmp_path, monkeypatch: pytest.MonkeyPatch) -> str:
    config_dir = str(tmp_path)
    monkeypatch.setenv("NAT_CONFIG_DIR", config_dir)
    # Settings.from_file() sets the class level directory, so record it here to have it restored afterwards. Resetting
    # the global settings makes the `global_settings` fixture load them from config_dir instead of the user config.
    monkeypatch.setattr(Settings, "_configuration_directory", config_dir, raising=False)
    monkeypatch.setattr(GlobalSettings, "_global_settings", None)
    return config_dir


@pytest.mark.usefixtures("config_dir")
def test_saved_channel_tokens_survive_reload(rest_registry_channel: dict, global_settings: Settings):
    channels = rest_registry_channel["channels"]
    channels["other_channel"] = {**channels["rest_channel"], "token": "other-token"}

    global_settings.update_settings(rest_registry_channel)

    loaded = Settings.from_file()
    assert get_secret_value(loaded.channels["rest_channel"].token) == "test-token"
    assert get_secret_value(loaded.channels["other_channel"].token) == "other-token"

    # Saving again for an unrelated change must not touch the remaining token
    remove_channel(channel_name="other_channel")

    loaded = Settings.from_file()
    assert list(loaded.channels) == ["rest_channel"]
    assert get_secret_value(loaded.channels["rest_channel"].token) == "test-token"


@pytest.mark.usefixtures("config_dir")
def test_print_channel_settings_masks_token(rest_registry_channel: dict,
                                            global_settings: Settings,
                                            caplog: pytest.LogCaptureFixture):
    global_settings.update_settings(rest_registry_channel)

    with caplog.at_level(logging.INFO, logger="nat.settings.global_settings"):
        global_settings.print_channel_settings()

    assert "rest_channel" in caplog.text
    assert "**********" in caplog.text
    assert "test-token" not in caplog.text
