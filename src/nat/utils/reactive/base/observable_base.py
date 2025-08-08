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
from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from typing import Generic
from typing import TypeVar

from aiq.utils.reactive.base.observer_base import ObserverBase
from aiq.utils.reactive.subscription import Subscription

# Covariant type param: An Observable producing type X can also produce
# a subtype of X.
_T_out_co = TypeVar("_T_out_co", covariant=True)  # pylint: disable=invalid-name
_T = TypeVar("_T")  # pylint: disable=invalid-name

OnNext = Callable[[_T], None]
OnError = Callable[[Exception], None]
OnComplete = Callable[[], None]


class ObservableBase(Generic[_T_out_co], ABC):
    """
    Abstract base class for an Observable that can be subscribed to.
    Produces items of type _T_out for its subscribers.
    """

    @typing.overload
    def subscribe(self, on_next: ObserverBase[_T_out_co]) -> Subscription:
        ...

    @typing.overload
    def subscribe(self,
                  on_next: OnNext[_T_out_co] | None = None,
                  on_error: OnError | None = None,
                  on_complete: OnComplete | None = None) -> Subscription:
        ...

    @abstractmethod
    def subscribe(self,
                  on_next: ObserverBase[_T_out_co] | OnNext[_T_out_co] | None = None,
                  on_error: OnError | None = None,
                  on_complete: OnComplete | None = None) -> Subscription:
        """
        Subscribes an Observer or callbacks to this Observable.

        If an Observer is provided, it will be subscribed to this Observable.
        If callbacks are provided, they will be wrapped into an Observer and
        subscribed to this Observable.
        """
        pass
