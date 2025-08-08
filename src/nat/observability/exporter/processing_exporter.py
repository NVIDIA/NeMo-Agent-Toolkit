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

import asyncio
import logging
from abc import abstractmethod
from collections.abc import Coroutine
from typing import Any
from typing import Generic
from typing import TypeVar

from nat.builder.context import ContextState
from nat.data_models.intermediate_step import IntermediateStep
from nat.observability.exporter.base_exporter import BaseExporter
from nat.observability.mixin.type_introspection_mixin import TypeIntrospectionMixin
from nat.observability.processor.callback_processor import CallbackProcessor
from nat.observability.processor.processor import Processor
from nat.utils.type_utils import DecomposedType
from nat.utils.type_utils import override

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

    Key Features:
    - Processor pipeline management (add, remove, clear)
    - Type compatibility validation between processors
    - Pipeline processing with error handling
    - Automatic type validation before export
    """

    def __init__(self, context_state: ContextState | None = None):
        """Initialize the processing exporter.

        Args:
            context_state: The context state to use for the exporter.
        """
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
            try:
                if not issubclass(processor.input_class, self._processors[-1].output_class):
                    raise ValueError(f"Processor {processor.__class__.__name__} input type {processor.input_type} "
                                     f"is not compatible with the {self._processors[-1].__class__.__name__} "
                                     f"output type {self._processors[-1].output_type}")
            except TypeError:
                # Handle cases where input_class or output_class are generic types that can't be used with issubclass
                # Fall back to type comparison for generic types
                logger.warning(
                    "Cannot use issubclass() for type compatibility check between "
                    "%s (%s) and %s (%s). Skipping compatibility check.",
                    processor.__class__.__name__,
                    processor.input_type,
                    self._processors[-1].__class__.__name__,
                    self._processors[-1].output_type)
        self._processors.append(processor)

        # Set up pipeline continuation callback for processors that support it
        if isinstance(processor, CallbackProcessor):
            # Create a callback that continues processing through the rest of the pipeline
            async def pipeline_callback(item):
                await self._continue_pipeline_after(processor, item)

            processor.set_done_callback(pipeline_callback)

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
            try:
                if not issubclass(first_processor.input_class, self.input_class):
                    raise ValueError(f"Processor {first_processor.__class__.__name__} input type "
                                     f"{first_processor.input_type} is not compatible with the "
                                     f"{self.input_type} input type")
            except TypeError as e:
                # Handle cases where classes are generic types that can't be used with issubclass
                logger.warning(
                    "Cannot validate type compatibility between %s (%s) "
                    "and exporter (%s): %s. Skipping validation.",
                    first_processor.__class__.__name__,
                    first_processor.input_type,
                    self.input_type,
                    e)

            # Validate that the last processor's output type is compatible with the exporter's output type
            try:
                if not DecomposedType.is_type_compatible(last_processor.output_type, self.output_type):
                    raise ValueError(f"Processor {last_processor.__class__.__name__} output type "
                                     f"{last_processor.output_type} is not compatible with the "
                                     f"{self.output_type} output type")
            except TypeError as e:
                # Handle cases where classes are generic types that can't be used with issubclass
                logger.warning(
                    "Cannot validate type compatibility between %s (%s) "
                    "and exporter (%s): %s. Skipping validation.",
                    last_processor.__class__.__name__,
                    last_processor.output_type,
                    self.output_type,
                    e)

    async def _process_pipeline(self, item: PipelineInputT) -> PipelineOutputT:
        """Process item through all registered processors.

        Args:
            item (PipelineInputT): The item to process (starts as PipelineInputT, can transform to PipelineOutputT)

        Returns:
            PipelineOutputT: The processed item after running through all processors
        """
        return await self._process_through_processors(self._processors, item)  # type: ignore

    async def _process_through_processors(self, processors: list[Processor], item: Any) -> Any:
        """Process an item through a list of processors.

        Args:
            processors (list[Processor]): List of processors to run the item through
            item (Any): The item to process

        Returns:
            The processed item after running through all processors
        """
        processed_item = item
        for processor in processors:
            try:
                processed_item = await processor.process(processed_item)
            except Exception as e:
                logger.error("Error in processor %s: %s", processor.__class__.__name__, e, exc_info=True)
                # Continue with unprocessed item rather than failing
        return processed_item

    async def _export_final_item(self, processed_item: Any, raise_on_invalid: bool = False) -> None:
        """Export a processed item with proper type handling.

        Args:
            processed_item (Any): The item to export
            raise_on_invalid (bool): If True, raise ValueError for invalid types instead of logging warning
        """
        if isinstance(processed_item, list):
            if len(processed_item) > 0:
                await self.export_processed(processed_item)
            else:
                logger.debug("Skipping export of empty batch")
        elif isinstance(processed_item, self.output_class):
            await self.export_processed(processed_item)
        else:
            if raise_on_invalid:
                raise ValueError(f"Processed item {processed_item} is not a valid output type. "
                                 f"Expected {self.output_class} or list[{self.output_class}]")
            logger.warning("Processed item %s is not a valid output type for export", processed_item)

    async def _continue_pipeline_after(self, source_processor: Processor, item: Any) -> None:
        """Continue processing an item through the pipeline after a specific processor.

        This is used when processors (like BatchingProcessor) need to inject items
        back into the pipeline flow to continue through downstream processors.

        Args:
            source_processor (Processor): The processor that generated the item
            item (Any): The item to continue processing through the remaining pipeline
        """
        try:
            # Find the source processor's position
            try:
                source_index = self._processors.index(source_processor)
            except ValueError:
                logger.error("Source processor %s not found in pipeline", source_processor.__class__.__name__)
                return

            # Process through remaining processors (skip the source processor)
            remaining_processors = self._processors[source_index + 1:]
            processed_item = await self._process_through_processors(remaining_processors, item)

            # Export the final result
            await self._export_final_item(processed_item)

        except Exception as e:
            logger.error("Failed to continue pipeline processing after %s: %s",
                         source_processor.__class__.__name__,
                         e,
                         exc_info=True)

    async def _export_with_processing(self, item: PipelineInputT) -> None:
        """Export an item after processing it through the pipeline.

        Args:
            item: The item to export
        """
        try:
            # Then, run through the processor pipeline
            final_item: PipelineOutputT = await self._process_pipeline(item)

            # Handle different output types from batch processors
            if isinstance(final_item, list) and len(final_item) == 0:
                logger.debug("Skipping export of empty batch from processor pipeline")
                return

            await self._export_final_item(final_item, raise_on_invalid=True)

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
    async def export_processed(self, item: PipelineOutputT | list[PipelineOutputT]) -> None:
        """Export the processed item.

        This method must be implemented by concrete exporters to handle
        the actual export logic after the item has been processed through the pipeline.

        Args:
            item: The processed item to export (PipelineOutputT type)
        """
        pass

    def _create_export_task(self, coro: Coroutine):
        """Create task with minimal overhead but proper tracking."""
        if not self._running:
            logger.warning("%s: Attempted to create export task while not running", self.name)
            return

        try:
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        except Exception as e:
            logger.error("%s: Failed to create task: %s", self.name, e, exc_info=True)
            raise

    @override
    async def _cleanup(self):
        """Enhanced cleanup that shuts down all shutdown-aware processors.

        Each processor is responsible for its own cleanup, including routing
        any final batches through the remaining pipeline via their done callbacks.
        """
        # Shutdown all processors that support it
        shutdown_tasks = []
        for processor in getattr(self, '_processors', []):
            shutdown_method = getattr(processor, 'shutdown', None)
            if shutdown_method:
                logger.debug("Shutting down processor: %s", processor.__class__.__name__)
                shutdown_tasks.append(shutdown_method())

        if shutdown_tasks:
            try:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
                logger.debug("Successfully shut down %d processors", len(shutdown_tasks))
            except Exception as e:
                logger.error("Error shutting down processors: %s", e, exc_info=True)

        # Call parent cleanup
        await super()._cleanup()
