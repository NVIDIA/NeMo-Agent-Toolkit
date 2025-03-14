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
from typing import TypeVar

from aiq.utils.reactive.base.observable_base import ObservableBase
from aiq.utils.reactive.base.observer_base import ObserverBase
from aiq.utils.reactive.observer import Observer
from aiq.utils.reactive.subscription import Subscription

# Covariant type param: An Observable producing type X can also produce
# a subtype of X.
_T_out_co = TypeVar("_T_out_co", covariant=True)  # pylint: disable=invalid-name
_T = TypeVar("_T")  # pylint: disable=invalid-name

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class Observable(ObservableBase[_T_out_co]):
    """
    Concrete base Observable that implements subscribe, deferring actual hooking
    logic to _subscribe_core.
    """

    __slots__ = ()

    def _subscribe_core(self, observer: ObserverBase) -> Subscription:
        """
        By default, does nothing. Subclasses should override this to
        attach the observer to their emission logic.
        """
        raise NotImplementedError("Observable._subscribe_core must be implemented by subclasses")

    @typing.override
    def subscribe(self,
                  on_next: ObserverBase[_T_out_co] | OnNext[_T_out_co] | None = None,
                  on_error: OnError | None = None,
                  on_complete: OnComplete | None = None) -> "Subscription":

        if isinstance(on_next, ObserverBase):
            return self._subscribe_core(on_next)

        return self._subscribe_core(Observer(on_next, on_error, on_complete))
