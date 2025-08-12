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

from nat.utils.reactive.observable import Observable
from nat.utils.reactive.observer import Observer


class MockObservable(Observable[str]):

    def __init__(self):
        super().__init__()
        self.observers = []

    def _subscribe_core(self, observer: Observer):
        # store the observer so we can emit manually
        self.observers.append(observer)
        # No real subscription logic here (like unsub). Could add if needed.

    def emit_value(self, val: str):
        for obs in self.observers:
            obs.on_next(val)

    def emit_error(self, exc: Exception):
        for obs in list(self.observers):
            obs.on_error(exc)

    def emit_complete(self):
        for obs in list(self.observers):
            obs.on_complete()


def test_observable_subscribe_observer():
    mock = MockObservable()
    items = []
    obs = Observer(on_next=items.append)
    mock.subscribe(obs)
    mock.emit_value("A")
    mock.emit_value("B")
    assert items == ["A", "B"]


def test_observable_subscribe_callbacks():
    mock = MockObservable()
    items = []
    errors = []

    def on_next_cb(x):
        items.append(x)  # pylint: disable=multiple-statements

    def on_err_cb(e):
        errors.append(str(e))  # pylint: disable=multiple-statements

    mock.subscribe(on_next_cb, on_err_cb)
    mock.emit_value("Hello")
    mock.emit_error(ValueError("Oops"))
    assert items == ["Hello"]
    assert errors == ["Oops"]
