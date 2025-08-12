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

import json

import pytest
from nat_user_report.user_report_tools import DeleteUserReportConfig
from nat_user_report.user_report_tools import GetUserReportConfig
from nat_user_report.user_report_tools import PutUserReportConfig
from nat_user_report.user_report_tools import UpdateUserReportConfig

from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.object_store import KeyAlreadyExistsError
from nat.data_models.object_store import NoSuchKeyError
from nat.object_store.in_memory_object_store import InMemoryObjectStoreConfig
from nat.object_store.models import ObjectStoreItem


@pytest.fixture
async def builder():
    """Pytest fixture to create a builder with an InMemoryObjectStore and user_report_tool functions."""
    async with WorkflowBuilder() as builder:
        await builder.add_object_store("test_object_store", InMemoryObjectStoreConfig())
        await builder.add_function(
            "get_user_report",
            GetUserReportConfig(object_store=ObjectStoreRef("test_object_store"), description="Get user report"))
        await builder.add_function(
            "put_user_report",
            PutUserReportConfig(object_store=ObjectStoreRef("test_object_store"), description="Put user report"))
        await builder.add_function(
            "update_user_report",
            UpdateUserReportConfig(object_store=ObjectStoreRef("test_object_store"), description="Update user report"))
        await builder.add_function(
            "delete_user_report",
            DeleteUserReportConfig(object_store=ObjectStoreRef("test_object_store"), description="Delete user report"))
        yield builder


@pytest.fixture
async def object_store(builder):
    """Pytest fixture to create an object store client."""
    return await builder.get_object_store_client("test_object_store")


@pytest.fixture
async def get_fn(builder):
    """Pytest fixture to get a function from the builder."""
    return builder.get_function("get_user_report")


@pytest.fixture
async def put_fn(builder):
    """Pytest fixture to get a function from the builder."""
    return builder.get_function("put_user_report")


@pytest.fixture
async def update_fn(builder):
    """Pytest fixture to get a function from the builder."""
    return builder.get_function("update_user_report")


@pytest.fixture
async def delete_fn(builder):
    """Pytest fixture to get a function from the builder."""
    return builder.get_function("delete_user_report")


class TestUserReportTools:
    """Test suite for user report tools using InMemoryObjectStore."""

    # Tests for get_user_report function
    async def test_get_user_report_valid_case(self, object_store, get_fn):
        """Test get_user_report with existing report."""
        # Setup: put a report in the object store
        test_report = {"user": "test_user", "data": "test data"}
        await object_store.put_object(
            "/reports/test_user/latest.json",
            ObjectStoreItem(data=json.dumps(test_report).encode("utf-8"), content_type="application/json"))

        # Test: get the report
        result = await get_fn.ainvoke(get_fn.input_schema(user_id="test_user"))
        assert result == json.dumps(test_report)

    async def test_get_user_report_with_date(self, object_store, get_fn):
        """Test get_user_report with specific date."""
        # Setup: put a report with specific date
        test_report = {"user": "test_user", "date": "2024-01-01"}
        await object_store.put_object(
            "/reports/test_user/2024-01-01.json",
            ObjectStoreItem(data=json.dumps(test_report).encode("utf-8"), content_type="application/json"))

        # Test: get the report with date
        result = await get_fn.ainvoke(get_fn.input_schema(user_id="test_user", date="2024-01-01"))
        assert result == json.dumps(test_report)

    async def test_get_user_report_not_found(self, get_fn):
        """Test get_user_report when report doesn't exist."""
        with pytest.raises(NoSuchKeyError):
            await get_fn.ainvoke(get_fn.input_schema(user_id="nonexistent_user"))

    # Tests for put_user_report function

    async def test_put_user_report_valid_case(self, object_store, put_fn):
        """Test put_user_report with new report."""
        test_report = json.dumps({"user": "test_user", "data": "new data"})
        result = await put_fn.ainvoke(put_fn.input_schema(report=test_report, user_id="test_user"))

        assert result == "User report for test_user with date latest added successfully"

        # Verify the report was stored
        stored_item = await object_store.get_object("/reports/test_user/latest.json")
        assert stored_item.data.decode("utf-8") == test_report

    async def test_put_user_report_with_date(self, object_store, put_fn):
        """Test put_user_report with specific date."""
        test_report = json.dumps({"user": "test_user", "date": "2024-01-01"})
        result = await put_fn.ainvoke(put_fn.input_schema(report=test_report, user_id="test_user", date="2024-01-01"))
        assert result == "User report for test_user with date 2024-01-01 added successfully"

        stored_item = await object_store.get_object("/reports/test_user/2024-01-01.json")
        assert stored_item.data.decode("utf-8") == test_report

    async def test_put_user_report_already_exists(self, object_store, put_fn):
        """Test put_user_report when report already exists."""
        initial_report = json.dumps({"user": "test_user", "data": "initial"})
        await object_store.put_object(
            "/reports/test_user/latest.json",
            ObjectStoreItem(data=initial_report.encode("utf-8"), content_type="application/json"))

        test_report = json.dumps({"user": "test_user", "data": "duplicate"})
        with pytest.raises(KeyAlreadyExistsError):
            await put_fn.ainvoke(put_fn.input_schema(report=test_report, user_id="test_user"))

    # Tests for update_user_report function (upsert behavior)

    async def test_update_user_report_new_report(self, object_store, update_fn):
        """Test update_user_report creating a new report."""
        test_report = json.dumps({"user": "test_user", "data": "new data"})
        result = await update_fn.ainvoke(update_fn.input_schema(report=test_report, user_id="test_user"))
        assert result == "User report for test_user with date latest updated"

        stored_item = await object_store.get_object("/reports/test_user/latest.json")
        assert stored_item.data.decode("utf-8") == test_report

    async def test_update_user_report_existing_report(self, object_store, update_fn):
        """Test update_user_report updating an existing report."""
        initial_report = json.dumps({"user": "test_user", "data": "initial"})
        await object_store.put_object(
            "/reports/test_user/latest.json",
            ObjectStoreItem(data=initial_report.encode("utf-8"), content_type="application/json"))

        updated_report = json.dumps({"user": "test_user", "data": "updated"})
        result = await update_fn.ainvoke(update_fn.input_schema(report=updated_report, user_id="test_user"))
        assert result == "User report for test_user with date latest updated"

        stored_item = await object_store.get_object("/reports/test_user/latest.json")
        assert stored_item.data.decode("utf-8") == updated_report

    async def test_update_user_report_with_date(self, object_store, update_fn):
        """Test update_user_report with specific date."""
        test_report = json.dumps({"user": "test_user", "date": "2024-01-01"})
        result = await update_fn.ainvoke(
            update_fn.input_schema(report=test_report, user_id="test_user", date="2024-01-01"))
        assert result == "User report for test_user with date 2024-01-01 updated"

        stored_item = await object_store.get_object("/reports/test_user/2024-01-01.json")
        assert stored_item.data.decode("utf-8") == test_report

    # Tests for delete_user_report function

    async def test_delete_user_report_valid_case(self, object_store, delete_fn):
        """Test delete_user_report with existing report."""
        test_report = json.dumps({"user": "test_user", "data": "to delete"})
        await object_store.put_object(
            "/reports/test_user/latest.json",
            ObjectStoreItem(data=test_report.encode("utf-8"), content_type="application/json"))

        result = await delete_fn.ainvoke(delete_fn.input_schema(user_id="test_user"))
        assert result == "User report for test_user with date latest deleted"

        with pytest.raises(NoSuchKeyError):
            await object_store.get_object("/reports/test_user/latest.json")

    async def test_delete_user_report_with_date(self, object_store, delete_fn):
        """Test delete_user_report with specific date."""
        # Setup: put a report with specific date
        test_report = json.dumps({"user": "test_user", "date": "2024-01-01"})
        await object_store.put_object(
            "/reports/test_user/2024-01-01.json",
            ObjectStoreItem(data=test_report.encode("utf-8"), content_type="application/json"))

        result = await delete_fn.ainvoke(delete_fn.input_schema(user_id="test_user", date="2024-01-01"))
        assert result == "User report for test_user with date 2024-01-01 deleted"

        # Verify the report was deleted
        with pytest.raises(NoSuchKeyError):
            await object_store.get_object("/reports/test_user/2024-01-01.json")

    async def test_delete_user_report_not_found(self, delete_fn):
        """Test delete_user_report when report doesn't exist."""
        with pytest.raises(NoSuchKeyError):
            await delete_fn.ainvoke(delete_fn.input_schema(user_id="nonexistent_user"))

    # Integration tests

    async def test_integration_full_workflow(self, put_fn, get_fn, update_fn, delete_fn):
        """Integration test that exercises all four functions together."""
        # Test workflow: put -> get -> update -> get -> delete -> get (should fail)

        # 1. Put a new report
        initial_report = json.dumps({"user": "integration_user", "data": "initial"})
        put_result = await put_fn.ainvoke(put_fn.input_schema(report=initial_report, user_id="integration_user"))
        assert "added successfully" in put_result

        # 2. Get the report
        get_result = await get_fn.ainvoke(get_fn.input_schema(user_id="integration_user"))
        assert get_result == initial_report

        # 3. Update the report
        updated_report = json.dumps({"user": "integration_user", "data": "updated"})
        update_result = await update_fn.ainvoke(
            update_fn.input_schema(report=updated_report, user_id="integration_user"))
        assert "updated" in update_result

        # 4. Get the updated report
        get_result_2 = await get_fn.ainvoke(get_fn.input_schema(user_id="integration_user"))
        assert get_result_2 == updated_report

        # 5. Delete the report
        delete_result = await delete_fn.ainvoke(delete_fn.input_schema(user_id="integration_user"))
        assert "deleted" in delete_result

        # 6. Try to get the deleted report (should fail)
        with pytest.raises(NoSuchKeyError):
            await get_fn.ainvoke(get_fn.input_schema(user_id="integration_user"))
