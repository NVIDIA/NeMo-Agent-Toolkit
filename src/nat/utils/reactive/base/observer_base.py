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

from abc import ABC
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

# Contravariant type param: An Observer that can accept type X can also
# accept any supertype of X.
_T_in_contra = TypeVar("_T_in_contra", contravariant=True)


class ObserverBase(Generic[_T_in_contra], ABC):
    """
    Abstract base class for an Observer that can receive events of type _T_in.

    Once on_error or on_complete is called, the observer is considered stopped.
    """

    @abstractmethod
    def on_next(self, value: _T_in_contra) -> None:
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
