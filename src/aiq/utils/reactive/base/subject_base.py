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
from abc import abstractmethod
from collections.abc import Callable
from typing import TypeVar

from .observable_base import ObservableBase
from .observer_base import ObserverBase

if typing.TYPE_CHECKING:
    from aiq.utils.reactive.subscription import Subscription

T = TypeVar("T")

OnNext = Callable[[T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class SubjectBase(ObserverBase[T], ObservableBase[T]):
    """
    Minimal interface we expect from the Subject for unsubscribing logic.
    """

    @abstractmethod
    def _unsubscribe_observer(self, observer: object) -> None:
        pass

    @abstractmethod
    def subscribe(self,
                  on_next: ObserverBase[T] | OnNext[T] | None = None,
                  on_error: OnError | None = None,
                  on_complete: OnComplete | None = None) -> "Subscription":
        """
        Subscribes an Observer or callbacks to this Observable.

        If an Observer is provided, it will be subscribed to this Observable.
        If callbacks are provided, they will be wrapped into an Observer and
        subscribed to this Observable.
        """
        pass

    @abstractmethod
    def on_next(self, value: T) -> None:
        """
        Called when a new item is produced. If the observer is stopped,
        this call should be ignored or raise an error.
        """
        pass

    @abstractmethod
    def on_error(self, exc: Exception) -> None:
        """
        Called when the producer signals an unrecoverable error.
        After this call, the observer is stopped.
        """
        pass

    @abstractmethod
    def on_complete(self) -> None:
        """
        Called when the producer signals completion (no more items).
        After this call, the observer is stopped.
        """
        pass
