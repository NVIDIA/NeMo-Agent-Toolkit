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

from abc import abstractmethod
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import Generic
from typing import TypeVar

from aiq.observability.processor.processor import Processor

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class CallbackProcessor(Processor[InputT, OutputT], Generic[InputT, OutputT]):
    """Abstract base class for processors that support done callbacks.

    Processors inheriting from this class can register callbacks that are
    invoked when items are ready for further processing or export.
    """

    @abstractmethod
    def set_done_callback(self, callback: Callable[[Any], Awaitable[None]]) -> None:
        """Set a callback function to be invoked when items are processed.

        Args:
            callback (Callable[[Any], Awaitable[None]]): Function to call with processed items
        """
        pass
