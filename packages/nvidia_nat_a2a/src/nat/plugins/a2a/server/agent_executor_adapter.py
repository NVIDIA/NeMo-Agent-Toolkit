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
"""Adapter to bridge NAT workflows with A2A AgentExecutor interface.

This module implements a message-only A2A agent for Phase 1, providing stateless
request/response interactions without task lifecycle management.
"""

import logging

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue
from a2a.types import InternalError
from a2a.types import InvalidParamsError
from a2a.types import UnsupportedOperationError
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from nat.runtime.session import SessionManager

logger = logging.getLogger(__name__)


class NATWorkflowAgentExecutor(AgentExecutor):
    """Adapts NAT workflows to A2A AgentExecutor interface as a message-only agent.

    This adapter implements Phase 1 support for A2A integration, providing stateless
    message-based interactions. Each request is handled independently without maintaining
    conversation state or task lifecycle.

    Key characteristics:
    - Stateless: Each message is processed independently
    - Synchronous: Returns immediate responses (no long-running tasks)
    - Message-only: Returns Message objects, not Task objects
    - Concurrent: Uses SessionManager's semaphore for concurrency control

    Note: Multi-turn conversations and task-based interactions are deferred to Phase 5.
    """

    def __init__(self, session_manager: SessionManager):
        """Initialize the adapter with a NAT SessionManager.

        Args:
            session_manager: The SessionManager for handling workflow execution
                with concurrency control via semaphore
        """
        self.session_manager = session_manager
        logger.info("Initialized NATWorkflowAgentExecutor (message-only) for workflow: %s",
                    session_manager.workflow.config.workflow.type)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the NAT workflow and return a message response.

        This is a message-only implementation (Phase 1):
        1. Extracts the user query from the A2A message
        2. Runs the NAT workflow (stateless, no conversation history)
        3. Returns the result as a Message object (not a Task)

        For Phase 1, each message is handled independently with no state preservation
        between requests. The context_id and task_id from the A2A protocol are mapped
        to NAT's conversation_id and user_message_id for tracing purposes only.

        Args:
            context: The A2A request context containing the user message
            event_queue: Queue for sending the response message back to the client

        Raises:
            ServerError: If validation fails or workflow execution errors occur
        """
        # Validate the request
        error = self._validate_request(context)
        if error:
            raise ServerError(error=InvalidParamsError())

        # Extract query from the message
        query = context.get_user_input()
        if not query:
            logger.error("No user input found in context")
            raise ServerError(error=InvalidParamsError())

        # Extract IDs for tracing (stored but not used for state management in Phase 1)
        context_id = context.context_id
        task_id = context.task_id

        logger.info("Processing message-only request (context_id=%s, task_id=%s): %s", context_id, task_id, query[:100])

        try:
            # Run the NAT workflow using SessionManager for proper concurrency handling
            # Each message gets its own independent session (stateless)
            # TODO: Add support for user input callbacks and authentication in later phases
            async with self.session_manager.session() as session:
                async with session.run(query) as runner:
                    # Get the result as a string
                    response_text = await runner.result(to_type=str)

            logger.info("Workflow completed successfully (context_id=%s, task_id=%s)", context_id, task_id)

            # Create and send the response message (message-only pattern)
            response_message = new_agent_text_message(
                response_text,
                context_id=context_id,
                task_id=task_id,
            )
            await event_queue.enqueue_event(response_message)

        except Exception as e:
            logger.error("Error executing NAT workflow (context_id=%s, task_id=%s): %s",
                         context_id,
                         task_id,
                         e,
                         exc_info=True)

            # Send error message back to client
            error_message = new_agent_text_message(
                f"An error occurred while processing your request: {str(e)}",
                context_id=context_id,
                task_id=task_id,
            )
            await event_queue.enqueue_event(error_message)
            raise ServerError(error=InternalError()) from e

    def _validate_request(self, context: RequestContext) -> bool:
        """Validate the incoming request context.

        Args:
            context: The request context to validate

        Returns:
            True if validation fails, False if validation succeeds
        """
        # Basic validation - can be extended as needed
        if not context.message:
            logger.error("Request context has no message")
            return True

        return False

    async def cancel(
        self,
        _context: RequestContext,
        _event_queue: EventQueue,
    ) -> None:
        """Handle task cancellation requests.

        Not applicable for message-only agents in Phase 1. Cancellation is a task-based
        feature that will be implemented in Phase 5 along with long-running task support.

        Args:
            _context: The request context (unused in Phase 1)
            _event_queue: Event queue for sending updates (unused in Phase 1)

        Raises:
            ServerError: Always raises UnsupportedOperationError
        """
        logger.warning("Task cancellation requested but not supported in message-only mode (Phase 1)")
        raise ServerError(error=UnsupportedOperationError())
