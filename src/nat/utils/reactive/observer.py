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
from collections.abc import Callable
from typing import TypeVar

from nat.utils.reactive.base.observer_base import ObserverBase

logger = logging.getLogger(__name__)

# Contravariant type param: An Observer that can accept type X can also
# accept any supertype of X.
_T_in_contra = TypeVar("_T_in_contra", contravariant=True)
_T = TypeVar("_T")

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Observer(ObserverBase[_T_in_contra]):
    """
    Concrete Observer that wraps user-provided callbacks into an ObserverBase.
    """

    def __init__(
        self,
        on_next: OnNext | None = None,
        on_error: OnError | None = None,
        on_complete: OnComplete | None = None,
    ) -> None:
        self._on_next = on_next
        self._on_error = on_error
        self._on_complete = on_complete
        self._stopped = False

    def on_next(self, value: _T) -> None:
        if self._stopped:
            return
        if self._on_next is None:
            return
        try:
            self._on_next(value)
        except Exception as exc:
            # If the callback itself raises, treat that as an error
            self.on_error(exc)

    def on_error(self, exc: Exception) -> None:
        if not self._stopped:
            if self._on_error:
                try:
                    self._on_error(exc)
                except Exception as e:
                    logger.exception("Error in on_error callback: %s", e)

    def on_complete(self) -> None:
        if not self._stopped:
            self._stopped = True
            if self._on_complete:
                try:
                    self._on_complete()
                except Exception as e:
                    logger.exception("Error in on_complete callback: %s", e)
