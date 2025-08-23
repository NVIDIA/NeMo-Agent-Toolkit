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

import typing
from collections.abc import Callable
from typing import Generic
from typing import TypeVar

if typing.TYPE_CHECKING:
    from nat.utils.reactive.base.subject_base import SubjectBase

_T = TypeVar("_T")

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Subscription(Generic[_T]):
    """
    Represents a subscription to a Subject.
    Unsubscribing removes the associated observer from the Subject's subscriber list.
    """

    def __init__(self, subject: "SubjectBase", observer: object | None):  # noqa: F821
        self._subject = subject
        self._observer = observer
        self._unsubscribed = False

    def unsubscribe(self) -> None:
        """
        Stop receiving further events.
        """
        if not self._unsubscribed and self._observer is not None:
            self._subject._unsubscribe_observer(self._observer)
            self._observer = None
            self._unsubscribed = True
