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

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.object_store.models import ObjectStoreItem

logger = logging.getLogger(__name__)


class GetUserReportConfig(FunctionBaseConfig, name="get_user_report"):
    object_store: ObjectStoreRef
    description: str


@register_function(config_type=GetUserReportConfig)
async def get_user_report(config: GetUserReportConfig, builder: Builder):
    object_store = await builder.get_object_store_client(object_store_name=config.object_store)

    async def _inner(user_id: str, date: str | None = None) -> str:
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Fetching report from %s", key)
        item = await object_store.get_object(key=key)
        return item.data.decode("utf-8")

    yield FunctionInfo.from_fn(_inner, description=config.description)


class PutUserReportConfig(FunctionBaseConfig, name="put_user_report"):
    object_store: ObjectStoreRef
    description: str


@register_function(config_type=PutUserReportConfig)
async def put_user_report(config: PutUserReportConfig, builder: Builder):
    object_store = await builder.get_object_store_client(object_store_name=config.object_store)

    async def _inner(report: str, user_id: str, date: str | None = None) -> str:
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

    yield FunctionInfo.from_fn(_inner, description=config.description)


class UpdateUserReportConfig(FunctionBaseConfig, name="update_user_report"):
    object_store: ObjectStoreRef
    description: str


@register_function(config_type=UpdateUserReportConfig)
async def update_user_report(config: UpdateUserReportConfig, builder: Builder):
    object_store = await builder.get_object_store_client(object_store_name=config.object_store)

    async def _inner(report: str, user_id: str, date: str | None = None) -> str:
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Update or insert report into %s for user %s with date %s", key, user_id, date)
        await object_store.upsert_object(key=key,
                                         item=ObjectStoreItem(data=report.encode("utf-8"),
                                                              content_type="application/json"))
        return f"User report for {user_id} with date {date} updated"

    yield FunctionInfo.from_fn(_inner, description=config.description)


class DeleteUserReportConfig(FunctionBaseConfig, name="delete_user_report"):
    object_store: ObjectStoreRef
    description: str


@register_function(config_type=DeleteUserReportConfig)
async def delete_user_report(config: DeleteUserReportConfig, builder: Builder):
    object_store = await builder.get_object_store_client(object_store_name=config.object_store)

    async def _inner(user_id: str, date: str | None = None) -> str:
        date = date or "latest"
        key = f"reports/{user_id}/{date}.json"
        logger.info("Delete report from %s for user %s with date %s", key, user_id, date)
        await object_store.delete_object(key=key)
        return f"User report for {user_id} with date {date} deleted"

    yield FunctionInfo.from_fn(_inner, description=config.description)
