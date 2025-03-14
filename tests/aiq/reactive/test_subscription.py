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

from aiq.utils.reactive.observer import Observer
from aiq.utils.reactive.subscription import Subscription


class MockSubjectBase:
    """
    Minimal stand-in for a subject that can be unsubscribed.
    """

    def __init__(self):
        self.unsubbed = None

    def _unsubscribe_observer(self, observer: object) -> None:
        self.unsubbed = observer


def test_subscription_unsubscribe():
    subject = MockSubjectBase()
    obs = Observer()  # not fully implemented, or we can do object() if we want
    sub = Subscription(subject, obs)

    assert sub._unsubscribed is False
    sub.unsubscribe()
    assert sub._unsubscribed is True
    assert subject.unsubbed == obs


def test_subscription_idempotent():
    subject = MockSubjectBase()
    obs = Observer()  # or just object()
    sub = Subscription(subject, obs)
    sub.unsubscribe()
    sub.unsubscribe()  # second unsubscribe does nothing

    assert subject.unsubbed == obs
    assert sub._unsubscribed is True
