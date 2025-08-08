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

from aiq.utils.reactive.observer import Observer

logger = logging.getLogger(__name__)


def test_observer_on_next():
    items = []
    obs = Observer(on_next=items.append)
    obs.on_next("Alpha")
    obs.on_next("Beta")
    assert items == ["Alpha", "Beta"]


def test_observer_on_error():
    errors = []

    def on_err(e):
        errors.append(str(e))

    obs = Observer(on_error=on_err)
    obs.on_error(ValueError("Something bad"))
    # further on_next calls do nothing
    obs.on_next("ignored")
    assert errors == ["Something bad"]


def test_observer_on_complete():
    completed = []
    obs = Observer(on_complete=lambda: completed.append("done"))
    obs.on_next("hello")
    obs.on_complete()
    # further on_next is ignored
    obs.on_next("ignored")
    assert completed == ["done"]


def test_observer_callback_raises():
    errors = []

    def fail_callback(x):
        raise RuntimeError("CallbackFail")

    def handle_error(e):
        errors.append(str(e))

    obs = Observer(on_next=fail_callback, on_error=handle_error)
    obs.on_next("test")
    assert errors == ["CallbackFail"]
