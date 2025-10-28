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

        Args:
            user_id (str): The user ID to check/create.
        """
        logger.info("Checking if Zep user exists")
        try:
            # Try to get the user
            await self._client.user.get(user_id=user_id)
            logger.info("Zep user already exists")
        except Exception:
            # User doesn't exist, create with basic info
            logger.info("Zep user not found, creating...")
            try:
                # Set realistic defaults for default_user
                if user_id == "default_user":
                    email = "jane.doe@example.com"
                    first_name = "Jane"
                    last_name = "Doe"
                else:
                    email = f"{user_id}@example.com"
                    first_name = "User"
                    last_name = user_id

                await self._client.user.add(user_id=user_id, email=email, first_name=first_name, last_name=last_name)
                logger.info("Successfully created Zep user")
            except Exception:
                # User might have been created by another request, ignore
                logger.warning("Error creating Zep user, assuming it exists")

    async def add_items(self, items: list[MemoryItem], **kwargs) -> None:
        """
        Insert Multiple MemoryItems into the memory using Zep v3 thread API.
        Each MemoryItem is translated and uploaded to a thread.
        Uses conversation_id from NAT context as thread_id for multi-thread support.

        Args:
            items (list[MemoryItem]): The items to be added.
            kwargs (dict): Provider-specific keyword arguments.
                - ignore_roles (list[str], optional): List of role types to ignore when adding
                  messages to graph memory. Available roles: norole, system, assistant, user,
                  function, tool.
        """
        # Extract Zep-specific parameters
        ignore_roles = kwargs.get("ignore_roles", None)

        coroutines = []

        # Iteratively insert memories into Zep using threads
        for memory_item in items:
            conversation = memory_item.conversation
            user_id = memory_item.user_id

            # Get thread_id from NAT context (unique per UI conversation)
            thread_id = Context.get().conversation_id

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
                # Create Message - role field instead of role_type in V3
                message = Message(content=msg["content"], role=msg["role"])
                messages.append(message)

            # Ensure thread exists - try to create it (more reliable than checking first)
            logger.info("Ensuring Zep thread exists")
            try:
                # Always try to create the thread
                await self._client.thread.create(thread_id=thread_id, user_id=user_id)
                logger.info("Successfully created new Zep thread")
            except Exception as create_error:
                # Thread likely already exists, which is fine
                error_msg = str(create_error).lower()
                if "already exists" in error_msg or "409" in error_msg or "conflict" in error_msg:
                    logger.info("Zep thread already exists (expected), continuing")
                else:
                    # Unexpected error, log it but continue anyway
                    logger.warning(
                        f"Unexpected error creating thread: {type(create_error).__name__}, attempting to continue")

            # Add messages to thread using Zep v3 API
            logger.info(f"Queueing add_messages for thread with {len(messages)} messages")

            # Build add_messages parameters
            add_messages_params = {"thread_id": thread_id, "messages": messages}
            if ignore_roles is not None:
                add_messages_params["ignore_roles"] = ignore_roles

            coroutines.append(self._client.thread.add_messages(**add_messages_params))

        await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:
        """
        Retrieve memory from Zep v3 using the high-level get_user_context API.
        Uses conversation_id from NAT context as thread_id for multi-thread support.

        Zep returns pre-formatted memory optimized for LLM consumption, including
        relevant facts, timestamps, and structured information from its knowledge graph.

        Args:
            query (str): The query string (not used by Zep's high-level API, included for interface compatibility).
            top_k (int): Maximum number of items to return (not used by Zep's context API).
            kwargs: Zep-specific keyword arguments.
                - user_id (str, required): The user ID for which to retrieve memory.
                - mode (str, optional): Retrieval mode. "basic" for fast retrieval (P95 < 200ms) or
                  "summary" for more comprehensive memory. Defaults to "basic".

        Returns:
            list[MemoryItem]: A single MemoryItem containing the formatted context from Zep.
        """
        user_id = kwargs.pop("user_id")  # Ensure user ID is in keyword arguments
        mode = kwargs.pop("mode", "basic")  # Get mode, default to "basic" for fast retrieval

        # Get thread_id from NAT context
        thread_id = Context.get().conversation_id

        # Fallback to default thread ID if no conversation_id available
        if not thread_id:
            thread_id = "default_zep_thread"

        try:
            # Use Zep v3 thread.get_user_context - returns pre-formatted context
            memory_response = await self._client.thread.get_user_context(thread_id=thread_id, mode=mode)
            context_string = memory_response.context or ""

            # Return as a single MemoryItem with the formatted context
            if context_string:
                return [
                    MemoryItem(conversation=[],
                               user_id=user_id,
                               memory=context_string,
                               metadata={
                                   "mode": mode, "thread_id": thread_id
                               })
                ]
            else:
                return []

        except Exception as e:
            # If thread doesn't exist or no context available, return empty list
            if "404" in str(e) or "not found" in str(e).lower():
                return []
            raise

    async def remove_items(self, **kwargs):
        """
        Remove items for a specific thread.

        Args:
            kwargs: Additional parameters.
                - thread_id (str): Thread ID to delete specific thread.
        """
        if "thread_id" in kwargs:
            thread_id = kwargs.pop("thread_id")
            await self._client.thread.delete(thread_id=thread_id)
        else:
            raise ValueError("thread_id not provided as part of the tool call.")
