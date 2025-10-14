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

from __future__ import annotations

import asyncio
import logging

from zep_cloud.client import AsyncZep
from zep_cloud.types import Message

from nat.builder.context import Context
from nat.memory.interfaces import MemoryEditor
from nat.memory.models import MemoryItem

logger = logging.getLogger(__name__)


class ZepEditor(MemoryEditor):
    """
    Wrapper class that implements NAT interfaces for Zep v3 Integrations Async.
    Uses thread-based memory management with automatic user creation.
    """

    def __init__(self, zep_client: AsyncZep):
        """
        Initialize class with Zep v3 AsyncZep Client.

        Args:
        zep_client (AsyncZep): Async client instance.
        """
        self._client = zep_client

    async def _ensure_user_exists(self, user_id: str) -> None:
        """
        Ensure a user exists in Zep v3, creating if necessary.

        Automatically creates users with information from Context (user_first_name,
        user_last_name, user_email) if available, otherwise uses fallback values.

        Args:
            user_id (str): The user ID to check/create.
        """
        logger.info("Checking if Zep user exists")
        try:
            # Try to get the user
            await self._client.user.get(user_id=user_id)
            logger.info("Zep user already exists")
        except Exception as e:
            # TODO: Replace with zep-cloud SDK-specific exceptions when available
            # User doesn't exist, create with info from Context or fallbacks
            logger.info("Zep user not found (%s), creating...", type(e).__name__)

            # Get user info from Context with fallbacks
            ctx = Context.get()
            first_name = ctx.user_first_name or "John"
            last_name = ctx.user_last_name or "Doe"
            email = ctx.user_email or "john.doe@example.com"

            try:
                await self._client.user.add(user_id=user_id, email=email, first_name=first_name, last_name=last_name)
                logger.info("Successfully created Zep user")
            except Exception as create_error:
                # TODO: Replace with zep-cloud SDK-specific exceptions when available
                # User might have been created by another request, ignore
                logger.warning("Error creating Zep user: %s, assuming it exists", type(create_error).__name__)

    async def add_items(self, items: list[MemoryItem], user_id: str, **kwargs) -> None:
        """
        Insert Multiple MemoryItems into the memory using Zep v3 thread API.
        Each MemoryItem is translated and uploaded to a thread.
        Uses conversation_id from NAT context as thread_id for multi-thread support.

        Args:
            items (list[MemoryItem]): The items to be added.
            user_id (str): The user ID for which to add memories.
            kwargs (dict): Provider-specific keyword arguments.
                - ignore_roles (list[str], optional): List of role types to ignore when adding
                  messages to graph memory. Available roles: norole, system, assistant, user,
                  function, tool.
        """
        # Warn if using default user_id
        if user_id == "default_NAT_user":
            logger.warning("No user_id provided in Context, using default value")

        # Get user info from Context for message names and check for warnings
        nat_context = Context.get()

        user_first_name = nat_context.user_first_name
        if not user_first_name:
            logger.warning("No user_first_name provided in Context, using default value 'John' for message attribution")
            user_first_name = "John"

        user_last_name = nat_context.user_last_name
        if not user_last_name:
            logger.warning("No user_last_name provided in Context, using default value 'Doe' for message attribution")
            user_last_name = "Doe"

        user_email = nat_context.user_email
        if not user_email:
            logger.warning("No user_email provided in Context, using default value 'john.doe@example.com'")
            user_email = "john.doe@example.com"

        user_full_name = f"{user_first_name} {user_last_name}"

        # Extract Zep-specific parameters
        ignore_roles = kwargs.get("ignore_roles", None)

        coroutines = []

        # Iteratively insert memories into Zep using threads
        for memory_item in items:
            conversation = memory_item.conversation

            # Get thread_id from NAT context (unique per UI conversation)
            nat_context = Context.get()
            thread_id = nat_context.conversation_id

            # Fallback to default thread ID if no conversation_id available
            if not thread_id:
                thread_id = "default_zep_thread"

            messages = []

            # Ensure user exists before creating thread
            await self._ensure_user_exists(user_id)

            # Skip if no conversation data
            if not conversation:
                continue

            for msg in conversation:
                # Create Message with name field for better graph construction
                # Use full name from Context for user messages, or explicit name if provided
                default_name = user_full_name if msg["role"] == "user" else "AI Assistant"
                message = Message(content=msg["content"], role=msg["role"], name=msg.get("name", default_name))
                messages.append(message)

            # Ensure thread exists - try to create it (more reliable than checking first)
            logger.info("Ensuring Zep thread exists")
            try:
                # Always try to create the thread
                await self._client.thread.create(thread_id=thread_id, user_id=user_id)
                logger.info("Successfully created new Zep thread")
            except Exception as create_error:
                # TODO: Replace with zep-cloud SDK-specific exceptions when available
                # Thread likely already exists, which is fine
                error_msg = str(create_error).lower()
                if "already exists" in error_msg or "409" in error_msg or "conflict" in error_msg:
                    logger.info("Zep thread already exists (expected), continuing")
                else:
                    # Unexpected error, log it but continue anyway
                    logger.warning("Unexpected error creating thread: %s, attempting to continue", type(create_error).__name__)

            # Add messages to thread using Zep v3 API
            logger.info("Queueing add_messages for thread with %d messages", len(messages))

            # Build add_messages parameters
            add_messages_params = {"thread_id": thread_id, "messages": messages}
            if ignore_roles is not None:
                add_messages_params["ignore_roles"] = ignore_roles

            coroutines.append(self._client.thread.add_messages(**add_messages_params))

        await asyncio.gather(*coroutines)

    async def retrieve_memory(self, query: str, user_id: str, **kwargs) -> str:
        """
        Retrieve formatted memory from Zep v3 using the high-level get_user_context API.
        Uses conversation_id from NAT context as thread_id for multi-thread support.

        Zep returns pre-formatted memory optimized for LLM consumption, including
        relevant facts, timestamps, and structured information from its knowledge graph.

        Args:
            query (str): The query string (not used by Zep's high-level API, included for interface compatibility).
            user_id (str): The user ID for which to retrieve memory.
            kwargs: Zep-specific keyword arguments.
                - mode (str, optional): Retrieval mode. "basic" for fast retrieval (P95 < 200ms) or
                  "summary" for more comprehensive memory. Defaults to "basic".

        Returns:
            str: Formatted memory string from Zep, or empty string if no memory available.
        """
        # Warn if using default user_id
        if user_id == "default_NAT_user":
            logger.warning("No user_id provided in Context, using default value")

        mode = kwargs.pop("mode", "basic")  # Get mode, default to "basic" for fast retrieval

        # Get thread_id from NAT context
        thread_id = Context.get().conversation_id

        # Fallback to default thread ID if no conversation_id available
        if not thread_id:
            thread_id = "default_zep_thread"

        try:
            # Use Zep v3 thread.get_user_context - returns pre-formatted context
            memory_response = await self._client.thread.get_user_context(thread_id=thread_id, mode=mode)
            # Return Zep's formatted context string directly
            return memory_response.context or ""

        except Exception as e:
            # If thread doesn't exist or no context available, return empty string
            if "404" in str(e) or "not found" in str(e).lower():
                return ""
            raise

    async def remove_items(self, user_id: str, **kwargs):
        """
        Remove items for a specific user.

        Args:
            user_id (str): The user ID for which to remove memories (not currently used by Zep).
            kwargs: Additional parameters.
                - thread_id (str): Optional thread ID to delete specific thread.
        """
        # Warn if using default user_id
        if user_id == "default_NAT_user":
            logger.warning("No user_id provided in Context, using default value")

        if "thread_id" in kwargs:
            thread_id = kwargs.pop("thread_id")
            await self._client.thread.delete(thread_id=thread_id)
        else:
            raise ValueError("thread_id not provided as part of the tool call.")
