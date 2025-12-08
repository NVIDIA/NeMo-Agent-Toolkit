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

from zep_cloud import ApiError
from zep_cloud import NotFoundError
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

    def __init__(self, zep_client: AsyncZep) -> None:
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
        logger.debug("Checking if Zep user exists")
        try:
            await self._client.user.get(user_id=user_id)
            logger.debug("Zep user already exists")
        except NotFoundError:
            # User doesn't exist, create with basic info
            logger.info("Zep user not found, creating...")
            try:
                # Set defaults only for default_user, otherwise use just user_id
                if user_id == "default_user":
                    email = "jane.doe@example.com"
                    first_name = "Jane"
                    last_name = "Doe"
                    await self._client.user.add(user_id=user_id,
                                                email=email,
                                                first_name=first_name,
                                                last_name=last_name)
                else:
                    # For non-default users, just use user_id (email/names not required)
                    await self._client.user.add(user_id=user_id)

                logger.info("Created Zep user")
            except ApiError as e:
                # Check if user was created by another request (409 Conflict)
                if e.response_data and e.response_data.get("status_code") == 409:
                    logger.info("Zep user already exists (409), continuing")
                else:
                    logger.error("Failed creating Zep user: %s", str(e))  # noqa: TRY400
                    raise
        except ApiError as e:
            logger.error("Failed fetching Zep user: %s", str(e))  # noqa: TRY400
            raise

    async def add_items(self, items: list[MemoryItem], **kwargs) -> None:
        """
        Insert Multiple MemoryItems into the memory using Zep v3 thread API.
        Each MemoryItem is translated and uploaded to a thread.
        Uses conversation_id from NAT context as thread_id for multi-thread support.

        Args:
            items (list[MemoryItem]): The items to be added.
            kwargs (dict): Provider-specific keyword arguments.

                - ignore_roles (list[str], optional): List of role types to ignore when adding
                  messages to graph memory. Available roles: system, assistant, user,
                  function, tool.
        """
        # Extract Zep-specific parameters
        ignore_roles = kwargs.get("ignore_roles", None)

        coroutines = []
        created_threads: set[str] = set()
        ensured_users: set[str] = set()

        # Iteratively insert memories into Zep using threads
        for memory_item in items:
            conversation = memory_item.conversation
            user_id = memory_item.user_id or "default_user"  # Validate user_id

            # Get thread_id from NAT context (unique per UI conversation)
            thread_id = Context.get().conversation_id

            # Fallback to default thread ID if no conversation_id available
            if not thread_id:
                thread_id = "default_zep_thread"

            messages = []

            # Ensure user exists before creating thread (only once per user)
            if user_id not in ensured_users:
                await self._ensure_user_exists(user_id)
                ensured_users.add(user_id)

            # Skip if no conversation data
            if not conversation:
                continue

            for msg in conversation:
                # Create Message - role field instead of role_type in V3
                message = Message(content=msg["content"], role=msg["role"])
                messages.append(message)

            # Ensure thread exists once per thread_id
            thread_ready = True
            if thread_id not in created_threads:
                logger.info("Ensuring Zep thread exists (thread_id=%s)", thread_id)
                try:
                    await self._client.thread.create(thread_id=thread_id, user_id=user_id)
                    logger.info("Created Zep thread (thread_id=%s)", thread_id)
                    created_threads.add(thread_id)
                except ApiError as create_error:
                    if create_error.response_data and create_error.response_data.get("status_code") == 409:
                        logger.info("Zep thread already exists (thread_id=%s)", thread_id)
                        created_threads.add(thread_id)
                    else:
                        logger.exception("Thread create failed (thread_id=%s)", thread_id)
                        thread_ready = False

            # Skip this item if thread creation failed unexpectedly
            if not thread_ready:
                continue

            # Add messages to thread using Zep v3 API
            logger.info("Queueing add_messages (thread_id=%s, count=%d)", thread_id, len(messages))

            # Build add_messages parameters
            add_messages_params = {"thread_id": thread_id, "messages": messages}
            if ignore_roles is not None:
                add_messages_params["ignore_roles"] = ignore_roles

            coroutines.append(self._client.thread.add_messages(**add_messages_params))

        await asyncio.gather(*coroutines)

    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[MemoryItem]:  # noqa: ARG002
        """
        Retrieve memory from Zep v3 using the high-level get_user_context API.
        Uses conversation_id from NAT context as thread_id for multi-thread support.

        Zep returns pre-formatted memory optimized for LLM consumption, including
        relevant facts, timestamps, and structured information from its knowledge graph.

        Args:
            query (str): The query string (not used by Zep's high-level API, included for interface compatibility).
            top_k (int): Maximum number of items to return (not used by Zep's context API).
            kwargs: Zep-specific keyword arguments.

                - user_id (str, required for response construction): Used only to construct the
                  returned MemoryItem. Zep v3's thread.get_user_context() only requires thread_id.
                - mode (str, optional): Retrieval mode. Zep server default is "summary". This
                  implementation uses mode="basic" (NAT's default) for performance (P95 < 200ms).
                  "summary" provides more comprehensive memory at the cost of latency.

        Returns:
            list[MemoryItem]: A single MemoryItem containing the formatted context from Zep.
        """
        # Validate required kwargs
        if "user_id" not in kwargs or not kwargs["user_id"]:
            raise ValueError("user_id is required.")
        user_id = kwargs.pop("user_id")
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

        except NotFoundError:
            # Thread doesn't exist or no context available
            return []
        except ApiError as e:
            logger.error("get_user_context failed (thread_id=%s): %s", thread_id, str(e))  # noqa: TRY400
            raise

    async def remove_items(self, **kwargs) -> None:
        """
        Remove memory items based on provided criteria.

        Supports two deletion modes:

        1. Delete a specific thread by thread_id
        2. Delete all threads for a user by user_id

        Args:
            kwargs: Additional parameters.

                - thread_id (str, optional): Thread ID to delete a specific thread.
                - user_id (str, optional): User ID to delete all threads for that user.
        """
        if "thread_id" in kwargs:
            # Delete specific thread
            thread_id = kwargs.pop("thread_id")
            logger.info("Deleting thread (thread_id=%s)", thread_id)
            await self._client.thread.delete(thread_id=thread_id)
        elif "user_id" in kwargs:
            # Delete all threads for a user
            user_id = kwargs.pop("user_id")
            logger.debug("Deleting all threads for user (user_id=%s)", user_id)

            # Get all threads for this user
            threads = await self._client.user.get_threads(user_id=user_id)
            logger.debug("Found %d threads for user (user_id=%s)", len(threads), user_id)

            # Delete each thread
            delete_coroutines = []
            for thread in threads:
                if thread.thread_id:
                    logger.debug("Queueing deletion of thread (thread_id=%s)", thread.thread_id)
                    delete_coroutines.append(self._client.thread.delete(thread_id=thread.thread_id))

            if delete_coroutines:
                await asyncio.gather(*delete_coroutines)
                logger.info("Deleted %d threads for user", len(delete_coroutines))
        else:
            raise ValueError("Either thread_id or user_id is required.")
