# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import AsyncGenerator
from typing import Optional
from typing import TypeVar

from nat.observability.context import ObservabilityContext

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nat.builder.workflow import Workflow

InputT = TypeVar("InputT")
SingleOutputT = TypeVar("SingleOutputT")
StreamingOutputT = TypeVar("StreamingOutputT")


class ObservabilityWorkflowInvoker:
    """
    Utility class for invoking workflows with observability context propagation.

    This class provides helper methods to invoke workflows while properly
    propagating and managing observability context across workflow boundaries.
    """

    @staticmethod
    async def invoke_workflow_with_context(workflow: "Workflow[InputT, StreamingOutputT, SingleOutputT]",
                                           message: InputT,
                                           workflow_name: str,
                                           parent_context: Optional[ObservabilityContext] = None,
                                           to_type: Optional[type] = None) -> SingleOutputT:
        """
        Invoke a workflow with observability context propagation.

        Args:
            workflow: The workflow to invoke
            message: The input message for the workflow
            workflow_name: The name of the workflow for observability tracking
            parent_context: Optional parent observability context
            to_type: Optional type to convert the result to

        Returns:
            The result of the workflow execution
        """
        # Create or propagate observability context
        obs_context = None
        if parent_context:
            obs_context = parent_context.create_child_context(workflow_name)
        else:
            # Check if there's a context in the current execution
            try:
                # Import here to avoid circular import
                from nat.builder.context import Context
                current_context = Context.get()
                existing_obs_context = current_context.observability_context
                if existing_obs_context:
                    obs_context = existing_obs_context.create_child_context(workflow_name)
                else:
                    obs_context = ObservabilityContext.create_root_context(workflow_name)
            except Exception:
                obs_context = ObservabilityContext.create_root_context(workflow_name)

        # Update workflow metadata with timing
        if obs_context:
            current_workflow = obs_context.get_current_workflow()
            if current_workflow:
                current_workflow.start_time = time.time()

        try:
            async with workflow.run(message, observability_context=obs_context) as runner:
                result = await runner.result(to_type=to_type)

                # Update workflow metadata on completion
                if obs_context:
                    current_workflow = obs_context.get_current_workflow()
                    if current_workflow:
                        current_workflow.end_time = time.time()
                        current_workflow.status = "completed"

                return result

        except Exception as e:
            # Update workflow metadata on failure and log error
            if obs_context:
                current_workflow = obs_context.get_current_workflow()
                if current_workflow:
                    current_workflow.end_time = time.time()
                    current_workflow.status = "failed"
                    current_workflow.tags["error"] = str(e)

            logger.error(f"Workflow '{workflow_name}' failed with error: {e}", exc_info=True)
            raise

    @staticmethod
    async def invoke_workflow_stream_with_context(
            workflow: "Workflow[InputT, StreamingOutputT, SingleOutputT]",
            message: InputT,
            workflow_name: str,
            parent_context: Optional[ObservabilityContext] = None,
            to_type: Optional[type] = None) -> AsyncGenerator[StreamingOutputT, None]:
        """
        Invoke a workflow with streaming output and observability context propagation.

        Args:
            workflow: The workflow to invoke
            message: The input message for the workflow
            workflow_name: The name of the workflow for observability tracking
            parent_context: Optional parent observability context
            to_type: Optional type to convert the result to

        Yields:
            The streaming output from the workflow
        """
        # Create or propagate observability context
        obs_context = None
        if parent_context:
            obs_context = parent_context.create_child_context(workflow_name)
        else:
            # Check if there's a context in the current execution
            try:
                # Import here to avoid circular import
                from nat.builder.context import Context
                current_context = Context.get()
                existing_obs_context = current_context.observability_context
                if existing_obs_context:
                    obs_context = existing_obs_context.create_child_context(workflow_name)
                else:
                    obs_context = ObservabilityContext.create_root_context(workflow_name)
            except Exception:
                obs_context = ObservabilityContext.create_root_context(workflow_name)

        # Update workflow metadata with timing
        if obs_context:
            current_workflow = obs_context.get_current_workflow()
            if current_workflow:
                current_workflow.start_time = time.time()

        try:
            async with workflow.run(message, observability_context=obs_context) as runner:
                async for item in runner.result_stream(to_type=to_type):
                    yield item

                # Update workflow metadata on completion
                if obs_context:
                    current_workflow = obs_context.get_current_workflow()
                    if current_workflow:
                        current_workflow.end_time = time.time()
                        current_workflow.status = "completed"

        except Exception as e:
            # Update workflow metadata on failure and log error
            if obs_context:
                current_workflow = obs_context.get_current_workflow()
                if current_workflow:
                    current_workflow.end_time = time.time()
                    current_workflow.status = "failed"
                    current_workflow.tags["error"] = str(e)

            logger.error(f"Streaming workflow '{workflow_name}' failed with error: {e}", exc_info=True)
            raise

    @staticmethod
    def get_current_observability_context() -> Optional[ObservabilityContext]:
        """
        Get the current observability context from the execution context.

        Returns:
            The current observability context if available, None otherwise
        """
        try:
            # Import here to avoid circular import
            from nat.builder.context import Context
            current_context = Context.get()
            return current_context.observability_context
        except Exception:
            return None

    @staticmethod
    def create_observability_context(workflow_name: str, trace_id: Optional[str] = None) -> ObservabilityContext:
        """
        Create a new observability context and set it in the current execution context.

        Args:
            workflow_name: The name of the root workflow
            trace_id: Optional trace ID to use

        Returns:
            The created observability context
        """
        context = ObservabilityContext.create_root_context(workflow_name, trace_id)

        # Set it in the current context if available
        try:
            # Import here to avoid circular import
            from nat.builder.context import Context
            current_context = Context.get()
            current_context.set_observability_context(context)
        except Exception:
            pass  # Context might not be available

        return context

    @staticmethod
    async def invoke_with_steps_and_context(workflow: "Workflow[InputT, StreamingOutputT, SingleOutputT]",
                                            message: InputT,
                                            workflow_name: str,
                                            parent_context: Optional[ObservabilityContext] = None,
                                            to_type: Optional[type] = None) -> tuple[SingleOutputT, Any]:
        """
        Invoke a workflow with intermediate steps and observability context propagation.

        Args:
            workflow: The workflow to invoke
            message: The input message for the workflow
            workflow_name: The name of the workflow for observability tracking
            parent_context: Optional parent observability context
            to_type: Optional type to convert the result to

        Returns:
            Tuple of (result, intermediate_steps)
        """
        # Create or propagate observability context
        obs_context = None
        if parent_context:
            obs_context = parent_context.create_child_context(workflow_name)
        else:
            # Check if there's a context in the current execution
            try:
                # Import here to avoid circular import
                from nat.builder.context import Context
                current_context = Context.get()
                existing_obs_context = current_context.observability_context
                if existing_obs_context:
                    obs_context = existing_obs_context.create_child_context(workflow_name)
                else:
                    obs_context = ObservabilityContext.create_root_context(workflow_name)
            except Exception:
                obs_context = ObservabilityContext.create_root_context(workflow_name)

        # Update workflow metadata with timing
        if obs_context:
            current_workflow = obs_context.get_current_workflow()
            if current_workflow:
                current_workflow.start_time = time.time()

        try:
            result, steps = await workflow.result_with_steps(
                message,
                to_type=to_type,
                observability_context=obs_context
            )

            # Update workflow metadata on completion
            if obs_context:
                current_workflow = obs_context.get_current_workflow()
                if current_workflow:
                    current_workflow.end_time = time.time()
                    current_workflow.status = "completed"

            return result, steps

        except Exception as e:
            # Update workflow metadata on failure and log error
            if obs_context:
                current_workflow = obs_context.get_current_workflow()
                if current_workflow:
                    current_workflow.end_time = time.time()
                    current_workflow.status = "failed"
                    current_workflow.tags["error"] = str(e)

            logger.error(f"Workflow with steps '{workflow_name}' failed with error: {e}", exc_info=True)
            raise
