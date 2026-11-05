# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Test text extraction from A2A tasks that did not complete."""

import pytest
from a2a.types import Task
from a2a.types import TaskState
from a2a.types import TaskStatus
from a2a.utils import new_agent_text_message

from nat.plugins.a2a.client.client_base import A2ABaseClient


def _task(state: TaskState, text: str | None = None) -> Task:
    message = new_agent_text_message(text) if text is not None else None
    return Task(id="task-1", context_id="ctx-1", status=TaskStatus(state=state, message=message))


class TestExtractTextFromTask:
    """Test the text returned to the caller for tasks that are not completed."""

    @pytest.fixture(name="client")
    def fixture_client(self) -> A2ABaseClient:
        return A2ABaseClient(base_url="http://localhost:10000")

    def test_failed_task_returns_status_message_text(self, client: A2ABaseClient):
        task = _task(TaskState.failed, "Upstream rate service is down")

        assert client.extract_text_from_events([(task, None)]) == "Task failed: Upstream rate service is down"

    def test_failed_task_without_message(self, client: A2ABaseClient):
        task = _task(TaskState.failed)

        assert client.extract_text_from_task(task) == "Task failed: Unknown error"

    @pytest.mark.parametrize("state, expected",
                             [(TaskState.rejected, "Task rejected: I can only convert currency."),
                              (TaskState.canceled, "Task canceled: I can only convert currency.")])
    def test_rejected_or_canceled_task_returns_status_message_text(self,
                                                                   client: A2ABaseClient,
                                                                   state: TaskState,
                                                                   expected: str):
        task = _task(state, "I can only convert currency.")

        assert client.extract_text_from_events([(task, None)]) == expected

    @pytest.mark.parametrize("state", [TaskState.rejected, TaskState.canceled])
    def test_rejected_or_canceled_task_without_message(self, client: A2ABaseClient, state: TaskState):
        task = _task(state)

        assert client.extract_text_from_task(task) == f"Task {state.value}"

    @pytest.mark.parametrize("state", [TaskState.input_required, TaskState.auth_required])
    def test_follow_up_question_is_returned(self, client: A2ABaseClient, state: TaskState):
        task = _task(state, "Which currency do you want to convert to?")

        assert client.extract_text_from_events([(task, None)]) == "Which currency do you want to convert to?"

    def test_input_required_without_message(self, client: A2ABaseClient):
        task = _task(TaskState.input_required)

        assert client.extract_text_from_task(task) == "Task in progress (state: input-required)"

    def test_working_task_reports_state(self, client: A2ABaseClient):
        task = _task(TaskState.working, "Looking up rates")

        assert client.extract_text_from_task(task) == "Task in progress (state: working)"
