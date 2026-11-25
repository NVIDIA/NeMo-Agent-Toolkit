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


@pytest.fixture(name="update_env")
def update_env_fixture(monkeypatch: pytest.MonkeyPatch):
    channel = SimpleNamespace(static_type=lambda: "rest")
    settings = SimpleNamespace(channels={"main": channel}, channel_names=["main"])
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


def test_update_channel_reprompts_until_a_configured_name(update_env, monkeypatch: pytest.MonkeyPatch):
    settings, configured = update_env
    _feed_inputs(monkeypatch, ["", "nope", "main"])

    gs.update_channel_interactive("typo")

    assert configured == ["main"]
    # No channel is created as a side effect of the lookup
    assert list(settings.channels) == ["main"]


def test_update_channel_with_known_name_does_not_prompt(update_env, monkeypatch: pytest.MonkeyPatch):
    _, configured = update_env
    _feed_inputs(monkeypatch, [])

    gs.update_channel_interactive("main")

    assert configured == ["main"]
