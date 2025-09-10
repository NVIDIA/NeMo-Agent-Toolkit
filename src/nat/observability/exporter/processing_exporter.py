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
    - Configurable None filtering: processors returning None can drop items from pipeline
    - Automatic type validation before export
    """
    # All ProcessingExporter instances automatically use this for signature checking
    _signature_method = '_process_pipeline'

    def __init__(self, context_state: ContextState | None = None, drop_nones: bool = True):
        """Initialize the processing exporter.

        Args:
            context_state (ContextState | None): The context state to use for the exporter.
            drop_nones (bool): Whether to drop items when processors return None (default: True).
        """
        super().__init__(context_state)
        self._processors: list[Processor] = []  # List of processors that implement process(item) -> item
        self._processor_names: dict[str, int] = {}  # Maps processor names to their positions
        self._pipeline_locked: bool = False  # Prevents modifications after startup
        self._drop_nones: bool = drop_nones  # Whether to drop None values between processors

    def add_processor(self,
                      processor: Processor,
                      name: str | None = None,
                      position: int | None = None,
                      before: str | None = None,
                      after: str | None = None) -> None:
        """Add a processor to the processing pipeline.

        Processors are executed in the order they are added. Processes can transform between any types (T -> U).
        Supports flexible positioning using names, positions, or relative placement.

        Args:
            processor (Processor): The processor to add to the pipeline
            name (str | None): Name for the processor (for later reference). Must be unique.
            position (int | None): Specific position to insert at (0-based index, -1 for append)
            before (str | None): Insert before the named processor
            after (str | None): Insert after the named processor

        Raises:
            RuntimeError: If pipeline is locked (after startup)
            ValueError: If positioning arguments conflict or named processor not found
        """
        self._check_pipeline_locked()

        # Determine insertion position
        insert_position = self._calculate_insertion_position(position, before, after)

        # Validate type compatibility at insertion point
        self._validate_insertion_compatibility(processor, insert_position)

        # Pre-validate name (no side effects yet)
        if name is not None:
            if not isinstance(name, str):
                raise TypeError(f"Processor name must be a string, got {type(name).__name__}")
            if name in self._processor_names:
                raise ValueError(f"Processor name '{name}' already exists")

        # Shift existing name positions (do this before list mutation)
        for proc_name, pos in list(self._processor_names.items()):
            if pos >= insert_position:
                self._processor_names[proc_name] = pos + 1

        # Insert the processor
        if insert_position == len(self._processors):
            self._processors.append(processor)
        else:
            self._processors.insert(insert_position, processor)

        # Record the new processor name, if provided
        if name is not None:
            self._processor_names[name] = insert_position

        # Set up pipeline continuation callback for processors that support it
        if isinstance(processor, CallbackProcessor):
            # Create a callback that continues processing through the rest of the pipeline
            async def pipeline_callback(item):
                await self._continue_pipeline_after(processor, item)

            processor.set_done_callback(pipeline_callback)

    def remove_processor(self, processor: Processor | str | int) -> None:
        """Remove a processor from the processing pipeline.

        Args:
            processor (Processor | str | int): The processor to remove (by name, position, or object).

        Raises:
            RuntimeError: If pipeline is locked (after startup)
            ValueError: If named processor or position not found
            TypeError: If processor argument has invalid type
        """
        self._check_pipeline_locked()

        # Determine processor and position to remove
        if isinstance(processor, str):
            # Remove by name
            if processor not in self._processor_names:
                raise ValueError(f"Processor '{processor}' not found in pipeline")
            position = self._processor_names[processor]
            processor_obj = self._processors[position]
        elif isinstance(processor, int):
            # Remove by position
            if not (0 <= processor < len(self._processors)):
                raise ValueError(f"Position {processor} is out of range [0, {len(self._processors) - 1}]")
            position = processor
            processor_obj = self._processors[position]
        elif isinstance(processor, Processor):
            # Remove by object (existing behavior)
            if processor not in self._processors:
                return  # Silently ignore if not found (existing behavior)
            position = self._processors.index(processor)
            processor_obj = processor
        else:
            raise TypeError(f"Processor must be a Processor object, string name, or int position, "
                            f"got {type(processor).__name__}")

        # Remove the processor
        self._processors.remove(processor_obj)

        # Remove from name mapping and update positions
        name_to_remove = None
        for name, pos in self._processor_names.items():
            if pos == position:
                name_to_remove = name
                break

        if name_to_remove:
            del self._processor_names[name_to_remove]

        # Update positions for processors that shifted
        for name, pos in self._processor_names.items():
            if pos > position:
                self._processor_names[name] = pos - 1

    def clear_processors(self) -> None:
        """Clear all processors from the pipeline."""
        self._check_pipeline_locked()
        self._processors.clear()
        self._processor_names.clear()

    def reset_pipeline(self) -> None:
        """Reset the pipeline to allow modifications.

        This unlocks the pipeline and clears all processors, allowing
        the pipeline to be reconfigured. Can only be called when the
        exporter is stopped.

        Raises:
            RuntimeError: If exporter is currently running
        """
        if self._running:
            raise RuntimeError("Cannot reset pipeline while exporter is running. "
                               "Call stop() first, then reset_pipeline().")

        self._pipeline_locked = False
        self._processors.clear()
        self._processor_names.clear()
        logger.debug("Pipeline reset - unlocked and cleared all processors")

    def get_processor_by_name(self, name: str) -> Processor | None:
        """Get a processor by its name.

        Args:
            name (str): The name of the processor to retrieve

        Returns:
            Processor | None: The processor with the given name, or None if not found
        """
        if not isinstance(name, str):
            raise TypeError(f"Processor name must be a string, got {type(name).__name__}")
        if name in self._processor_names:
            position = self._processor_names[name]
            return self._processors[position]
        logger.debug("Processor '%s' not found in pipeline", name)
        return None

    def _check_pipeline_locked(self) -> None:
        """Check if pipeline is locked and raise error if it is."""
        if self._pipeline_locked:
            raise RuntimeError("Cannot modify processor pipeline after exporter has started. "
                               "Pipeline must be fully configured before calling start().")

    def _calculate_insertion_position(self, position: int | None, before: str | None, after: str | None) -> int:
        """Calculate the insertion position based on provided arguments.

        Args:
            position (int | None): Explicit position (0-based index, -1 for append)
            before (str | None): Insert before this named processor
            after (str | None): Insert after this named processor

        Returns:
            int: The calculated insertion position

        Raises:
            ValueError: If arguments conflict or named processor not found
        """
        # Check for conflicting arguments
        args_provided = sum(x is not None for x in [position, before, after])
        if args_provided > 1:
            raise ValueError("Only one of position, before, or after can be specified")

        # Default to append
        if args_provided == 0:
            return len(self._processors)

        # Handle explicit position
        if position is not None:
            if position == -1:
                return len(self._processors)
            if 0 <= position <= len(self._processors):
                return position
            raise ValueError(f"Position {position} is out of range [0, {len(self._processors)}]")

        # Handle before/after named processors
        if before is not None:
            if not isinstance(before, str):
                raise TypeError(f"'before' parameter must be a string, got {type(before).__name__}")
            if before not in self._processor_names:
                raise ValueError(f"Processor '{before}' not found in pipeline")
            return self._processor_names[before]

        if after is not None:
            if not isinstance(after, str):
                raise TypeError(f"'after' parameter must be a string, got {type(after).__name__}")
            if after not in self._processor_names:
                raise ValueError(f"Processor '{after}' not found in pipeline")
            return self._processor_names[after] + 1

        # Should never reach here
        return len(self._processors)

    def _validate_insertion_compatibility(self, processor: Processor, position: int) -> None:
        """Validate type compatibility for processor insertion.

        Args:
            processor (Processor): The processor to insert
            position (int): The position where it will be inserted

        Raises:
            ValueError: If processor is not compatible with neighbors
        """
        # Check compatibility with neighbors
        if position > 0:
            predecessor = self._processors[position - 1]
            self._check_processor_compatibility(predecessor,
                                                processor,
                                                "predecessor",
                                                str(predecessor.output_type),
                                                str(processor.input_type))

        if position < len(self._processors):
            successor = self._processors[position]
            self._check_processor_compatibility(processor,
                                                successor,
                                                "successor",
                                                str(processor.output_type),
                                                str(successor.input_type))

    def _check_processor_compatibility(self,
                                       source_processor: Processor,
                                       target_processor: Processor,
                                       relationship: str,
                                       source_type: str,
                                       target_type: str) -> None:
        """Check type compatibility between two processors using Pydantic validation.

        Args:
            source_processor (Processor): The processor providing output
            target_processor (Processor): The processor receiving input
            relationship (str): Description of relationship ("predecessor" or "successor")
            source_type (str): String representation of source type
            target_type (str): String representation of target type
        """
        # Use Pydantic-based type compatibility checking
        if not source_processor.is_output_compatible_with(target_processor.input_type):
            raise ValueError(f"Processor {target_processor.__class__.__name__} input type {target_type} "
                             f"is not compatible with {relationship} {source_processor.__class__.__name__} "
                             f"output type {source_type}")

    async def _pre_start(self) -> None:

        # Validate that the pipeline is compatible with the exporter
        if len(self._processors) > 0:
            first_processor = self._processors[0]
            last_processor = self._processors[-1]

            # validate that the first processor's input type is compatible with the exporter's input type
            if not first_processor.is_compatible_with_input(self.input_type):
                logger.error("First processor %s input=%s incompatible with exporter input=%s",
                             first_processor.__class__.__name__,
                             first_processor.input_type,
                             self.input_type)
                raise ValueError("First processor incompatible with exporter input")
            # Validate that the last processor's output type is compatible with the exporter's output type
            # Use DecomposedType.is_type_compatible for the final export stage to allow batch compatibility
            # This enables BatchingProcessor[T] -> Exporter[T] patterns where the exporter handles both T and list[T]
            if not DecomposedType.is_type_compatible(last_processor.output_type, self.output_type):
                logger.error("Last processor %s output=%s incompatible with exporter output=%s",
                             last_processor.__class__.__name__,
                             last_processor.output_type,
                             self.output_type)
                raise ValueError("Last processor incompatible with exporter output")

        # Lock the pipeline to prevent further modifications
        self._pipeline_locked = True

    async def _process_pipeline(self, item: PipelineInputT) -> PipelineOutputT | None:
        """Process item through all registered processors.

        Args:
            item (PipelineInputT): The item to process (starts as PipelineInputT, can transform to PipelineOutputT)

        Returns:
            PipelineOutputT | None: The processed item after running through all processors
        """
        return await self._process_through_processors(self._processors, item)  # type: ignore

    async def _process_through_processors(self, processors: list[Processor], item: Any) -> Any:
        """Process an item through a list of processors.

        Args:
            processors (list[Processor]): List of processors to run the item through
            item (Any): The item to process

        Returns:
            Any: The processed item after running through all processors, or None if
                drop_nones is True and any processor returned None
        """
        processed_item = item
        for processor in processors:
            try:
                processed_item = await processor.process(processed_item)
                # Drop None values between processors if configured to do so
                if self._drop_nones and processed_item is None:
                    logger.debug("Processor %s returned None, dropping item from pipeline",
                                 processor.__class__.__name__)
                    return None
            except Exception as e:
                logger.exception("Error in processor %s: %s", processor.__class__.__name__, e)
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
        elif self.validate_output_type(processed_item):
            await self.export_processed(processed_item)
        else:
            if raise_on_invalid:
                logger.error("Invalid processed item type for export: %s (expected %s or list[%s])",
                             type(processed_item),
                             self.output_type,
                             self.output_type)
                raise ValueError("Invalid processed item type for export")
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
                logger.exception("Source processor %s not found in pipeline", source_processor.__class__.__name__)
                return

            # Process through remaining processors (skip the source processor)
            remaining_processors = self._processors[source_index + 1:]
            processed_item = await self._process_through_processors(remaining_processors, item)

            # Skip export if remaining pipeline dropped the item (returned None)
            if processed_item is None:
                logger.debug("Item was dropped by remaining processor pipeline, skipping export")
                return

            # Export the final result
            await self._export_final_item(processed_item)

        except Exception as e:
            logger.exception("Failed to continue pipeline processing after %s: %s",
                             source_processor.__class__.__name__,
                             e)

    async def _export_with_processing(self, item: PipelineInputT) -> None:
        """Export an item after processing it through the pipeline.

        Args:
            item (PipelineInputT): The item to export
        """
        try:
            # Then, run through the processor pipeline
            final_item: PipelineOutputT | None = await self._process_pipeline(item)

            # Skip export if pipeline dropped the item (returned None)
            if final_item is None:
                logger.debug("Item was dropped by processor pipeline, skipping export")
                return

            # Handle different output types from batch processors
            if isinstance(final_item, list) and len(final_item) == 0:
                logger.debug("Skipping export of empty batch from processor pipeline")
                return

            await self._export_final_item(final_item, raise_on_invalid=True)

        except Exception as e:
            logger.error("Failed to export item '%s': %s", item, e)
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
        if self.validate_input_type(event):
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
            item (PipelineOutputT | list[PipelineOutputT]): The processed item to export (PipelineOutputT type)
        """
        pass

    def _create_export_task(self, coro: Coroutine) -> None:
        """Create task with minimal overhead but proper tracking.

        Args:
            coro: The coroutine to create a task for
        """
        if not self._running:
            logger.warning("%s: Attempted to create export task while not running", self.name)
            return

        try:
            task = asyncio.create_task(coro)
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        except Exception as e:
            logger.error("%s: Failed to create task: %s", self.name, e)
            raise

    @override
    async def _cleanup(self) -> None:
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
                logger.exception("Error shutting down processors: %s", e)

        # Call parent cleanup
        await super()._cleanup()
