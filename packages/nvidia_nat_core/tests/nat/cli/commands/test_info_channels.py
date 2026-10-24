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
from click.testing import CliRunner

from nat.cli.commands.info.list_channels import list_channels
from nat.settings.global_settings import Settings


@pytest.mark.parametrize("channel_type, listed, not_listed", [("rest", "rest_channel", "local_channel"),
                                                              ("local", "local_channel", "rest_channel")])
def test_list_channels_filter_by_type(rest_registry_channel: dict,
                                      local_registry_channel: dict,
                                      global_settings: Settings,
                                      caplog: pytest.LogCaptureFixture,
                                      channel_type: str,
                                      listed: str,
                                      not_listed: str):
    channels = {**rest_registry_channel["channels"], **local_registry_channel["channels"]}
    # Set the channels in memory only, update_settings() would write them to the user config file
    global_settings.channels = global_settings.model_validate({"channels": channels}).channels

    with caplog.at_level(logging.INFO):
        result = CliRunner().invoke(list_channels, ["--type", channel_type])

    assert result.exit_code == 0
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert listed in caplog.text
    assert not_listed not in caplog.text
