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
from abc import abstractmethod
from collections.abc import Coroutine
from typing import Generic
from typing import TypeVar

from aiq.builder.context import AIQContextState
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.observability.exporter.base_exporter import BaseExporter
from aiq.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin
from aiq.observability.processor.processor import Processor
from aiq.utils.type_utils import override

PipelineInputT = TypeVar("PipelineInputT")
PipelineOutputT = TypeVar("PipelineOutputT")

logger = logging.getLogger(__name__)


class ProcessingExporter(Generic[PipelineInputT, PipelineOutputT], BaseExporter, TypeIntrospectionMixin):
    """A base class for telemetry exporters with processing pipeline support.

    This class extends BaseExporter to add processor pipeline functionality.
    It manages a chain of processors that can transform items before export.

    The generic types work as follows:
    - PipelineInputT: The type of items that enter the processing pipeline (e.g., Span)
    - PipelineOutputT: The type of items after processing through the pipeline (e.g., converted format)

    Features:
    - Processor pipeline management (add, remove, clear)
    - Type compatibility validation between processors
    - Pipeline processing with error handling
    - Automatic type validation before export

    Args:
        context_state (AIQContextState, optional): The context state to use for the exporter. Defaults to None.
    """

    # ProcessingExporter doesn't add any additional attributes that need isolation
    # The _processors list should be preserved (shared) across copies
    _isolate_attributes: set = set()

    def __init__(self, context_state: AIQContextState | None = None):
        super().__init__(context_state)
        self._processors: list[Processor] = []  # List of processors that implement process(item) -> item

    def add_processor(self, processor: Processor) -> None:
        """Add a processor to the processing pipeline.

        Processors are executed in the order they are added.
        Processors can transform between any types (T -> U).

        Args:
            processor: The processor to add to the pipeline
        """

        # Check if the processor is compatible with the last processor in the pipeline
        if len(self._processors) > 0:
            if not issubclass(processor.input_type, self._processors[-1].output_type):
                raise ValueError(
                    f"Processor {processor.__class__.__name__} input type {processor.input_type} is not compatible "
                    f"with the {self._processors[-1].__class__.__name__} output type {self._processors[-1].output_type}"
                )

        self._processors.append(processor)

    def remove_processor(self, processor: Processor) -> None:
        """Remove a processor from the processing pipeline.

        Args:
            processor: The processor to remove from the pipeline
        """
        if processor in self._processors:
            self._processors.remove(processor)

    def clear_processors(self) -> None:
        """Clear all processors from the pipeline."""
        self._processors.clear()

    async def _pre_start(self) -> None:
        if len(self._processors) > 0:
            first_processor = self._processors[0]
            last_processor = self._processors[-1]

            # validate that the first processor's input type is compatible with the exporter's input type
            if not issubclass(first_processor.input_type, self.input_type):
                raise ValueError(f"Processor {first_processor.__class__.__name__} input type "
                                 f"{first_processor.input_type} is not compatible with the "
                                 f"{self.input_type} input type")

            # validate that the last processor's output type is compatible with the exporter's output type
            if not issubclass(last_processor.output_type, self.output_type):
                raise ValueError(f"Processor {last_processor.__class__.__name__} output type "
                                 f"{last_processor.output_type} is not compatible with the "
                                 f"{self.output_type} output type")

    async def _process_pipeline(self, item: PipelineInputT) -> PipelineOutputT:
        """Process item through all registered processors.

        Args:
            item: The item to process (starts as PipelineInputT, can transform to PipelineOutputT)

        Returns:
            The processed item after running through all processors
        """
        processed_item = item
        for processor in self._processors:
            try:
                processed_item = await processor.process(processed_item)
            except Exception as e:
                logger.error("Error in processor %s: %s", processor.__class__.__name__, e, exc_info=True)
                # Continue with unprocessed item rather than failing the export

        return processed_item  # type: ignore

    async def _export_with_processing(self, item: PipelineInputT) -> None:
        """Export an item after processing it through the pipeline.

        Args:
            item: The item to export
        """
        try:
            # Then, run through the processor pipeline
            final_item: PipelineOutputT = await self._process_pipeline(item)
            if not isinstance(final_item, self.output_class):
                raise ValueError(f"Processed item {final_item} is not an instance of {self.output_class}")
            await self.export_processed(final_item)

        except Exception as e:
            logger.error("Failed to export item '%s': %s", item, e, exc_info=True)
            raise

    @override
    def export(self, event: IntermediateStep) -> None:
        """Export an IntermediateStep event through the processing pipeline.

        This method converts the IntermediateStep to the expected PipelineInputT type,
        processes it through the pipeline, and exports the result.

        Args:
            event (IntermediateStep): The event to be exported.
        """
        # Convert IntermediateStep to PipelineInputT and create export task
        if isinstance(event, self.input_class):
            input_item: PipelineInputT = event  # type: ignore
            coro = self._export_with_processing(input_item)
            self._create_export_task(coro)
        else:
            logger.warning("Event %s is not compatible with input type %s", event, self.input_type)

    @abstractmethod
    async def export_processed(self, item: PipelineOutputT) -> None:
        """Export the processed item.

        This method must be implemented by concrete exporters to handle
        the actual export logic after the item has been processed through the pipeline.

        Args:
            item: The processed item to export (PipelineOutputT type)
        """
        pass

    def _create_export_task(self, coro: Coroutine):
        """Create an export task for the given coroutine.

        Args:
            coro (Coroutine): The coroutine to run as a task.
        """
        if not self._running:
            logger.warning("%s: Attempted to create export task while not running", self.name)
            return

        # Submit the export task to the event loop and track it
        try:

            task = self._loop.create_task(coro)
            # Add a name to the task for better debugging
            task.set_name(f"{self.name}_export_task")
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception as e:
            logger.error("%s: Failed to create export task: %s", self.name, e, exc_info=True)
            raise
