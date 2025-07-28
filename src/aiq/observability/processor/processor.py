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

from aiq.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin

InputT = TypeVar('InputT')
OutputT = TypeVar('OutputT')


class Processor(Generic[InputT, OutputT], TypeIntrospectionMixin, ABC):
    """Generic protocol for processors that can convert between types in export pipelines.

    Processors are the building blocks of processing pipelines in exporters. They can
    transform data from one type to another, enabling flexible data processing chains.

    The generic types work as follows:
    - InputT: The type of items that this processor accepts
    - OutputT: The type of items that this processor produces

    Key Features:
    - Type-safe transformations through generics
    - Type introspection capabilities via TypeIntrospectionMixin
    - Async processing support
    - Chainable in processing pipelines

    Inheritance Structure:
    - Inherits from TypeIntrospectionMixin for type introspection capabilities
    - Implements Generic[InputT, OutputT] for type safety
    - Abstract base class requiring implementation of process()

    Example:
        .. code-block:: python

            class SpanToOtelProcessor(Processor[Span, OtelSpan]):
                async def process(self, item: Span) -> OtelSpan:
                    return convert_span_to_otel(item)

    Note:
        Processors are typically added to ProcessingExporter instances to create
        transformation pipelines. The exporter validates type compatibility between
        chained processors.
    """

    @abstractmethod
    async def process(self, item: InputT) -> OutputT:
        """Process an item and return a potentially different type.

        Args:
            item: The item to process
        """
        pass
