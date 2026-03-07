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
"""Regression tests for ToolTalk evaluator scoring logic.

These tests use pre-constructed conversations with specific prediction patterns
to verify that the evaluator produces deterministic, expected metric values.
No LLM or network access required.
"""

import json
import os

import pytest

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.benchmarks.tooltalk.evaluator import _evaluate_single


def _get_database_dir():
    try:
        import tooltalk
        db_dir = os.path.join(os.path.dirname(tooltalk.__file__), "data", "databases")
        return db_dir if os.path.isdir(db_dir) else None
    except ImportError:
        return None


@pytest.fixture
def database_dir():
    db_dir = _get_database_dir()
    if db_dir is None:
        pytest.skip("tooltalk not installed or database directory not found")
    return db_dir


def _make_alarm_conversation(predictions: list[dict]) -> dict:
    """Build a ToolTalk AddAlarm conversation with given predictions."""
    return {
        "name": "regression-test",
        "conversation_id": "regression-001",
        "suites_used": ["Alarm"],
        "apis_used": ["AddAlarm"],
        "scenario": "regression",
        "user": {
            "username": "justinkool",
            "email": "test@test.com",
            "phone": "123-456-7890",
            "name": "Justin Kool",
            "password": "testpass123",
            "session_token": "98a5a87a-7714-b404",
        },
        "metadata": {
            "location": "New York",
            "timestamp": "2023-09-11 13:00:00",
            "session_token": "98a5a87a-7714-b404",
            "username": "justinkool",
        },
        "conversation": [
            {"index": 0, "role": "user", "text": "Set an alarm for 6:30 PM"},
            {
                "index": 1,
                "role": "assistant",
                "text": "Alarm set.",
                "apis": [{
                    "request": {"api_name": "AddAlarm", "parameters": {
                        "session_token": "98a5a87a-7714-b404", "time": "18:30:00",
                    }},
                    "response": {"alarm_id": "5bff-dd80"},
                    "exception": None,
                }],
                "predictions": predictions,
            },
        ],
    }


class TestToolTalkEvaluatorRegression:
    """Regression tests: pinned metric values for known prediction patterns."""

    def test_single_correct_prediction_is_success(self, database_dir):
        """1 correct prediction matching 1 ground truth = success."""
        predictions = [
            {
                "role": "api",
                "request": {"api_name": "AddAlarm", "parameters": {
                    "session_token": "98a5a87a-7714-b404", "time": "18:30:00",
                }},
                "response": {"alarm_id": "5bff-dd80"},
                "exception": None,
            },
            {"role": "assistant", "text": "Alarm set."},
        ]
        conv = _make_alarm_conversation(predictions)
        item = EvalInputItem(
            id="reg-001", input_obj=json.dumps(conv),
            expected_output_obj=json.dumps(conv),
            output_obj=json.dumps(conv), full_dataset_entry=conv,
        )
        result = _evaluate_single(item, database_dir)

        # Pinned expected values
        assert result.score == 1.0
        assert result.reasoning["recall"] == 1.0
        assert result.reasoning["matches"] == 1
        assert result.reasoning["predictions"] == 1
        assert result.reasoning["bad_actions"] == 0
        assert result.reasoning["bad_action_rate"] == 0.0
        assert result.reasoning["success"] == 1.0

    def test_correct_plus_extra_calls_has_bad_actions(self, database_dir):
        """1 correct + 2 extra same calls = recall 1.0 but bad_action_rate > 0, success 0."""
        predictions = [
            {
                "role": "api",
                "request": {"api_name": "AddAlarm", "parameters": {
                    "session_token": "98a5a87a-7714-b404", "time": "18:30:00",
                }},
                "response": {"alarm_id": "5bff-dd80"},
                "exception": None,
            },
            {
                "role": "api",
                "request": {"api_name": "AddAlarm", "parameters": {
                    "session_token": "98a5a87a-7714-b404", "time": "18:30:00",
                }},
                "response": {"alarm_id": "aaaa-bbbb"},
                "exception": None,
            },
            {
                "role": "api",
                "request": {"api_name": "AddAlarm", "parameters": {
                    "session_token": "98a5a87a-7714-b404", "time": "18:30:00",
                }},
                "response": {"alarm_id": "cccc-dddd"},
                "exception": None,
            },
            {"role": "assistant", "text": "Alarm set."},
        ]
        conv = _make_alarm_conversation(predictions)
        item = EvalInputItem(
            id="reg-002", input_obj=json.dumps(conv),
            expected_output_obj=json.dumps(conv),
            output_obj=json.dumps(conv), full_dataset_entry=conv,
        )
        result = _evaluate_single(item, database_dir)

        assert result.reasoning["recall"] == 1.0
        assert result.reasoning["matches"] == 1
        assert result.reasoning["predictions"] == 3
        assert result.reasoning["actions"] == 3
        assert result.reasoning["bad_actions"] == 2
        assert result.reasoning["success"] == 0.0
        assert result.score == 0.0

    def test_wrong_api_call_zero_recall(self, database_dir):
        """Calling the wrong API = recall 0, success 0."""
        predictions = [
            {
                "role": "api",
                "request": {"api_name": "DeleteAlarm", "parameters": {
                    "session_token": "98a5a87a-7714-b404", "alarm_id": "5bff-dd80",
                }},
                "response": None,
                "exception": "Alarm not found",
            },
            {"role": "assistant", "text": "Done."},
        ]
        conv = _make_alarm_conversation(predictions)
        item = EvalInputItem(
            id="reg-003", input_obj=json.dumps(conv),
            expected_output_obj=json.dumps(conv),
            output_obj=json.dumps(conv), full_dataset_entry=conv,
        )
        result = _evaluate_single(item, database_dir)

        assert result.reasoning["recall"] == 0.0
        assert result.reasoning["matches"] == 0
        assert result.score == 0.0

    def test_no_predictions_zero_recall(self, database_dir):
        """No API predictions at all = recall 0."""
        predictions = [
            {"role": "assistant", "text": "I can't do that."},
        ]
        conv = _make_alarm_conversation(predictions)
        item = EvalInputItem(
            id="reg-004", input_obj=json.dumps(conv),
            expected_output_obj=json.dumps(conv),
            output_obj=json.dumps(conv), full_dataset_entry=conv,
        )
        result = _evaluate_single(item, database_dir)

        assert result.reasoning["recall"] == 0.0
        assert result.reasoning["predictions"] == 0
        assert result.reasoning["bad_action_rate"] == 0.0
        assert result.score == 0.0
