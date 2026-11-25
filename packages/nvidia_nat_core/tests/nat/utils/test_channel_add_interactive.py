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

from types import SimpleNamespace

import pytest

from nat.utils.settings import global_settings as gs


@pytest.fixture(name="add_env")
def add_env_fixture(monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(channels={"existing": {}})
    registry = SimpleNamespace(
        get_registered_channel_info_by_channel_type=lambda channel_type: SimpleNamespace(config_type=object))
    configured: list[str] = []

    monkeypatch.setattr(gs.GlobalSettings, "get", classmethod(lambda cls: settings))
    monkeypatch.setattr(gs.GlobalTypeRegistry, "get", classmethod(lambda cls: registry))
    monkeypatch.setattr(gs,
                        "configure_registry_channel",
                        lambda config_type, channel_name: configured.append(channel_name))
    return settings, configured


def _feed_inputs(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def test_add_channel_rejects_empty_and_whitespace_names(add_env, monkeypatch: pytest.MonkeyPatch):
    settings, configured = add_env
    _feed_inputs(monkeypatch, ["", "   ", "my_channel"])

    gs.add_channel_interative("rest")

    assert "" not in settings.channels
    assert "my_channel" in settings.channels
    assert configured == ["my_channel"]


def test_add_channel_rejects_existing_name(add_env, monkeypatch: pytest.MonkeyPatch):
    settings, configured = add_env
    _feed_inputs(monkeypatch, ["existing", "other"])

    gs.add_channel_interative("rest")

    assert configured == ["other"]
