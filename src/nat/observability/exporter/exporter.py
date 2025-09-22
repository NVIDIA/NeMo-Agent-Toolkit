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
from abc import ABC
from abc import abstractmethod
from collections.abc import AsyncGenerator

from nat.data_models.intermediate_step import IntermediateStep

logger = logging.getLogger(__name__)


class Exporter(ABC):

    @abstractmethod
    async def start(self) -> AsyncGenerator[None]:
        """Subscribes to event stream and starts the exporter.

        This is an async context manager that should be used with 'async with'.
        The exporter is automatically stopped when exiting the context.

        Usage::

            .. code-block:: python

                async with exporter.start():
                    # Exporter is now running and subscribed to events
                    # Your workflow code here
                    pass

        Note:
            Implementations should use the @asynccontextmanager decorator.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Unsubscribes to the event stream and stops the exporter."""
        pass

    @abstractmethod
    def export(self, event: IntermediateStep) -> None:
        """This method is called on each event from the event stream to initiate the trace export.

        Args:
            event (IntermediateStep): The event to be exported.
        """
        pass

    @abstractmethod
    def on_error(self, exc: Exception) -> None:
        """Handle an error in the event subscription.

        Args:
            exc (Exception): The error to handle.
        """
        pass

    @abstractmethod
    def on_complete(self) -> None:
        """Handle the completion of the event stream.

        This method is called when the event stream is complete.
        """
        pass
