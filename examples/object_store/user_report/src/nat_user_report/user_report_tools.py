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

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.function import FunctionGroup
from nat.cli.register_workflow import register_function_group
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.function import FunctionGroupBaseConfig
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)


class UserReportConfig(FunctionGroupBaseConfig, name="user_report"):
    """Configuration for the user report function group.

    This function group provides CRUD operations for user reports stored in an object store.
    All functions share the same object store reference and configuration.
    """
    object_store: ObjectStoreRef = Field(description="The object store to use for storing user reports")

    # Function descriptions
    get_description: str
    put_description: str
    update_description: str
    delete_description: str


@register_function_group(config_type=UserReportConfig)
async def user_report_group(config: UserReportConfig, builder: Builder):
    """Register a function group for user report operations.

    This function group demonstrates:
    1. Shared configuration across all CRUD operations
    2. Shared object store resource
    3. Individual function descriptions
    4. Consistent error handling and logging
    """
    # Get the shared object store client
    object_store = await builder.get_object_store_client(object_store_name=config.object_store)

    # Define the individual functions with shared object store access
    async def get(user_id: str, date: str | None = None) -> str:
        """Get a user report from the object store."""
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Fetching report from %s", key)
        item = await object_store.get_object(key=key)
        return item.data.decode("utf-8")

    async def put(report: str, user_id: str, date: str | None = None) -> str:
        """Store a new user report in the object store."""
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Putting new report into %s for user %s with date %s", key, user_id, date)
        try:
            await object_store.put_object(key=key,
                                          item=ObjectStoreItem(data=report.encode("utf-8"),
                                                               content_type="application/json"))
            return f"User report for {user_id} with date {date} added successfully"
        except KeyAlreadyExistsError:
            return f"User report for {user_id} with date {date} already exists"

    async def update(report: str, user_id: str, date: str | None = None) -> str:
        """Update or create a user report in the object store."""
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Update or insert report into %s for user %s with date %s", key, user_id, date)
        await object_store.upsert_object(key=key,
                                         item=ObjectStoreItem(data=report.encode("utf-8"),
                                                              content_type="application/json"))
        return f"User report for {user_id} with date {date} updated"

    async def delete(user_id: str, date: str | None = None) -> str:
        """Delete a user report from the object store."""
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Delete report from %s for user %s with date %s", key, user_id, date)
        await object_store.delete_object(key=key)
        return f"User report for {user_id} with date {date} deleted"

    group = FunctionGroup(config=config)
    group.add_function("get", get, description=config.get_description)
    group.add_function("put", put, description=config.put_description)
    group.add_function("update", update, description=config.update_description)
    group.add_function("delete", delete, description=config.delete_description)
    yield group
