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
"""Tests for the ToolTalk evaluator — no LLM required.

These tests construct conversations with predictions already in place
(simulating what the workflow would produce) and verify that the evaluator
correctly computes ToolTalk metrics.
"""

import json
import os

import pytest

from nat.data_models.evaluator import EvalInputItem
from nat.plugins.benchmarks.tooltalk.evaluator import _evaluate_single


def _get_database_dir():
    """Get the path to ToolTalk's database directory."""
    try:
        import tooltalk
        db_dir = os.path.join(os.path.dirname(tooltalk.__file__), "data", "databases")
        if os.path.isdir(db_dir):
            return db_dir
    except ImportError:
        pass
    return None


@pytest.fixture
def database_dir():
    db_dir = _get_database_dir()
    if db_dir is None:
        pytest.skip("tooltalk not installed or database directory not found")
    return db_dir


def _make_conversation_with_predictions(predictions_match_ground_truth: bool) -> dict:
    """Create a ToolTalk conversation with predictions already attached.

    If predictions_match_ground_truth is True, the predictions exactly match
    the ground truth API calls (success case).
    """
    correct_api_call = {
        "request": {
            "api_name": "AddAlarm",
            "parameters": {
                "session_token": "98a5a87a-7714-b404", "time": "18:30:00"
            },
        },
        "response": {
            "alarm_id": "5bff-dd80"
        },
        "exception": None,
    }

    wrong_api_call = {
        "request": {
            "api_name": "AddAlarm",
            "parameters": {
                "session_token": "98a5a87a-7714-b404", "time": "19:00:00"
            },
        },
        "response": {
            "alarm_id": "aaaa-bbbb"
        },
        "exception": None,
    }

    prediction_api = correct_api_call.copy() if predictions_match_ground_truth else wrong_api_call.copy()
    prediction_api["role"] = "api"
    prediction_api["match"] = False  # Will be set by evaluate_predictions
    prediction_api["bad_action"] = False

    return {
        "name":
            "test-conversation",
        "conversation_id":
            "test-001",
        "suites_used": ["Alarm"],
        "apis_used": ["AddAlarm"],
        "scenario":
            "test",
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
            {
                "index": 0, "role": "user", "text": "Set an alarm for 6:30 PM"
            },
            {
                "index": 1,
                "role": "assistant",
                "text": "Alarm set.",
                "apis": [correct_api_call],
                "predictions": [
                    prediction_api,
                    {
                        "role": "assistant", "text": "Alarm set."
                    },
                ],
            },
        ],
    }


class TestToolTalkEvaluator:

    def test_perfect_predictions_score_1(self, database_dir):
        """When predictions exactly match ground truth, success score should be 1.0."""
        conv = _make_conversation_with_predictions(predictions_match_ground_truth=True)

        item = EvalInputItem(
            id="test-001",
            input_obj=json.dumps(conv),
            expected_output_obj=json.dumps(conv),
            output_obj=json.dumps(conv),
            full_dataset_entry=conv,
        )

        result = _evaluate_single(item, database_dir)

        assert result.id == "test-001"
        assert result.score == 1.0
        assert result.reasoning["success"] == 1.0
        assert result.reasoning["recall"] == 1.0
        assert result.reasoning["bad_action_rate"] == 0.0

    def test_wrong_predictions_score_0(self, database_dir):
        """When predictions don't match ground truth, success should be 0."""
        conv = _make_conversation_with_predictions(predictions_match_ground_truth=False)

        item = EvalInputItem(
            id="test-002",
            input_obj=json.dumps(conv),
            expected_output_obj=json.dumps(conv),
            output_obj=json.dumps(conv),
            full_dataset_entry=conv,
        )

        result = _evaluate_single(item, database_dir)

        assert result.id == "test-002"
        assert result.score == 0.0
        assert result.reasoning["recall"] == 0.0

    def test_none_output_returns_zero_score(self, database_dir):
        """If output_obj is None (workflow failed), score should be 0."""
        item = EvalInputItem(
            id="test-003",
            input_obj="{}",
            expected_output_obj="{}",
            output_obj=None,
            full_dataset_entry={},
        )

        result = _evaluate_single(item, database_dir)

        assert result.score == 0.0
        assert "error" in result.reasoning
