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

import threading
from collections.abc import Callable
from typing import TypeVar

from aiq.utils.reactive.base.subject_base import SubjectBase
from aiq.utils.reactive.observable import Observable
from aiq.utils.reactive.observer import Observer
from aiq.utils.reactive.subscription import Subscription

T = TypeVar("T")

OnNext = Callable[[T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Subject(Observable[T], Observer[T], SubjectBase[T]):
    """
    A Subject is both an Observer (receives events) and an Observable (sends events).
    - Maintains a list of ObserverBase[T].
    - No internal buffering or replay; events are only delivered to current subscribers.
    - Thread-safe via a lock.

    Once on_error or on_complete is called, the Subject is closed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.RLock()
        self._closed = False
        self._error: Exception | None = None
        self._observers: list[Observer[T]] = []
        self._disposed = False

    # ==========================================================================
    # Observable[T] - for consumers
    # ==========================================================================
    def _subscribe_core(self, observer: Observer[T]) -> Subscription:
        """
        Subscribe to this subject. If disposed, returns a dummy subscription.
        Otherwise, registers the given observer.
        """
        with self._lock:
            if self._disposed:
                # Already disposed => no subscription
                return Subscription(self, None)

            self._observers.append(observer)
            return Subscription(self, observer)

    # ==========================================================================
    # ObserverBase[T] - for producers
    # ==========================================================================
    def on_next(self, value: T) -> None:
        """
        Called by producers to emit an item. Delivers synchronously to each observer.
        If closed or disposed, do nothing.
        """
        with self._lock:
            if self._closed or self._disposed:
                return
            # Copy the current observers to avoid mutation issues
            current_observers = list(self._observers)

        # Deliver outside the lock
        for obs in current_observers:
            obs.on_next(value)

    def on_error(self, exc: Exception) -> None:
        """
        Called by producers to signal an error. Notifies all observers.
        """
        with self._lock:
            if self._closed or self._disposed:
                return
            current_obs = list(self._observers)

        for obs in current_obs:
            obs.on_error(exc)

    def on_complete(self) -> None:
        """
        Called by producers to signal completion. Notifies all observers, then
        clears them. Subject is closed.
        """
        with self._lock:
            if self._closed or self._disposed:
                return
            current_observers = list(self._observers)
            self.dispose()

        for obs in current_observers:
            obs.on_complete()

    # ==========================================================================
    # SubjectBase - internal unsubscribing
    # ==========================================================================
    def _unsubscribe_observer(self, observer: Observer[T]) -> None:
        with self._lock:
            if not self._disposed and observer in self._observers:
                self._observers.remove(observer)

    # ==========================================================================
    # Disposal
    # ==========================================================================
    def dispose(self) -> None:
        """
        Immediately close the Subject. No future on_next, on_error, or on_complete.
        Clears all observers.
        """
        with self._lock:
            if not self._disposed:
                self._disposed = True
                self._observers.clear()
                self._closed = True
                self._error = None
