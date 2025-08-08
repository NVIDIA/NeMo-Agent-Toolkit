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

from nat.utils.reactive.observer import Observer
from nat.utils.reactive.subject import Subject


def test_subject_basic():
    sub = Subject[str]()

    items1, items2 = [], []
    obs1 = Observer(on_next=items1.append)
    obs2 = Observer(on_next=items2.append)

    sub1 = sub.subscribe(obs1)
    _ = sub.subscribe(obs2)

    sub.on_next("X")
    sub.on_next("Y")
    assert items1 == ["X", "Y"]
    assert items2 == ["X", "Y"]

    # Unsubscribe first
    sub1.unsubscribe()
    sub.on_next("Z")
    assert items1 == ["X", "Y"]
    assert items2 == ["X", "Y", "Z"]


def test_subject_error():
    sub = Subject[str]()
    errors = []
    obs = Observer(on_error=lambda e: errors.append(str(e)))

    sub.subscribe(obs)
    sub.on_error(ValueError("Err"))
    # subsequent events do nothing if we consider on_error closes the subject
    sub.on_next("ignored")
    assert errors == ["Err"]


def test_subject_complete():
    sub = Subject[str]()
    items = []
    obs = Observer(on_next=items.append)
    sub.subscribe(obs)
    sub.on_next("a")
    assert items == ["a"]
    sub.on_complete()

    # further items do nothing
    sub.on_next("b")
    assert items == ["a"]


def test_subject_dispose():
    sub = Subject[str]()
    items = []
    sub.subscribe(Observer(on_next=items.append))
    sub.on_next("One")
    assert items == ["One"]

    sub.dispose()
    sub.on_next("Two")
    assert items == ["One"]


def test_subject_late_subscriber_after_dispose():
    sub = Subject[str]()
    sub.dispose()
    items = []
    sub.subscribe(Observer(on_next=items.append))
    sub.on_next("ignored")
    assert not items
