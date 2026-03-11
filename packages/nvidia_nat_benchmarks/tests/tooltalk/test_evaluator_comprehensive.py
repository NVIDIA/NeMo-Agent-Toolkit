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
"""Comprehensive tests for ToolTalk evaluator — covers all metric fields and edge cases."""

import json
import os

import pytest

from nat.data_models.evaluator import EvalInputItem

try:
    import tooltalk
    _DB_DIR = os.path.join(os.path.dirname(tooltalk.__file__), "data", "databases")
    _HAS_TOOLTALK = os.path.isdir(_DB_DIR)
except ImportError:
    _HAS_TOOLTALK = False

pytestmark = pytest.mark.skipif(not _HAS_TOOLTALK, reason="tooltalk not installed")

from nat.plugins.benchmarks.tooltalk.evaluator import _evaluate_single  # noqa: E402


def _alarm_conv(predictions):
    """Build a standard AddAlarm conversation with given predictions."""
    return {
        "name":
            "test",
        "conversation_id":
            "test-001",
        "suites_used": ["Alarm"],
        "apis_used": ["AddAlarm"],
        "scenario":
            "test",
        "user": {
            "username": "testuser",
            "email": "t@t.com",
            "phone": "123-456-7890",
            "name": "Test",
            "password": "pass",
            "session_token": "tok-123"
        },
        "metadata": {
            "location": "NY", "timestamp": "2023-09-11 13:00:00", "session_token": "tok-123", "username": "testuser"
        },
        "conversation": [
            {
                "index": 0, "role": "user", "text": "Set alarm for 6:30"
            },
            {
                "index": 1,
                "role": "assistant",
                "text": "Done.",
                "apis": [{
                    "request": {
                        "api_name": "AddAlarm", "parameters": {
                            "session_token": "tok-123", "time": "18:30:00"
                        }
                    },
                    "response": {
                        "alarm_id": "5bff-dd80"
                    },
                    "exception": None
                }],
                "predictions": predictions
            },
        ],
    }


def _make_item(conv):
    return EvalInputItem(
        id="test-001",
        input_obj=json.dumps(conv),
        expected_output_obj=json.dumps(conv),
        output_obj=json.dumps(conv),
        full_dataset_entry=conv,
    )


class TestAllMetricFields:
    """Verify every metric field in the reasoning dict."""

    def test_perfect_match_all_fields(self):
        preds = [
            {
                "role": "api",
                "request": {
                    "api_name": "AddAlarm", "parameters": {
                        "session_token": "tok-123", "time": "18:30:00"
                    }
                },
                "response": {
                    "alarm_id": "5bff-dd80"
                },
                "exception": None
            },
            {
                "role": "assistant", "text": "Done."
            },
        ]
        result = _evaluate_single(_make_item(_alarm_conv(preds)), _DB_DIR)
        r = result.reasoning

        assert r["predictions"] == 1
        assert r["ground_truths"] == 1
        assert r["matches"] == 1
        assert r["recall"] == 1.0
        assert r["actions"] == 1
        assert r["valid_actions"] == 1
        assert r["bad_actions"] == 0
        assert r["action_precision"] == 1.0
        assert r["bad_action_rate"] == 0.0
        assert r["success"] == 1.0
        assert r["soft_success"] == 1.0

    def test_no_api_predictions_all_fields(self):
        preds = [{"role": "assistant", "text": "I can't do that."}]
        result = _evaluate_single(_make_item(_alarm_conv(preds)), _DB_DIR)
        r = result.reasoning

        assert r["predictions"] == 0
        assert r["ground_truths"] == 1
        assert r["matches"] == 0
        assert r["recall"] == 0.0
        assert r["bad_action_rate"] == 0.0
        assert r["success"] == 0.0

    def test_soft_success_calculation(self):
        """3 predictions: 1 match + 2 bad actions → soft_success = recall*(1-bad_rate)."""
        preds = [
            {
                "role": "api",
                "request": {
                    "api_name": "AddAlarm", "parameters": {
                        "session_token": "tok-123", "time": "18:30:00"
                    }
                },
                "response": {
                    "alarm_id": "5bff-dd80"
                },
                "exception": None
            },
            {
                "role": "api",
                "request": {
                    "api_name": "AddAlarm", "parameters": {
                        "session_token": "tok-123", "time": "18:30:00"
                    }
                },
                "response": {
                    "alarm_id": "aaaa-bbbb"
                },
                "exception": None
            },
            {
                "role": "api",
                "request": {
                    "api_name": "AddAlarm", "parameters": {
                        "session_token": "tok-123", "time": "18:30:00"
                    }
                },
                "response": {
                    "alarm_id": "cccc-dddd"
                },
                "exception": None
            },
            {
                "role": "assistant", "text": "Done."
            },
        ]
        result = _evaluate_single(_make_item(_alarm_conv(preds)), _DB_DIR)
        r = result.reasoning

        assert r["recall"] == 1.0
        assert r["bad_actions"] == 2
        assert r["bad_action_rate"] == pytest.approx(2 / 3)
        assert r["soft_success"] == pytest.approx(1.0 * (1 - 2 / 3))
        assert r["success"] == 0.0  # bad_action_rate > 0


class TestMalformedInput:
    """Edge cases for malformed evaluator input."""

    def test_invalid_json_output(self):
        item = EvalInputItem(
            id="bad",
            input_obj="{}",
            expected_output_obj="{}",
            output_obj="not valid json at all",
            full_dataset_entry={},
        )
        result = _evaluate_single(item, _DB_DIR)
        assert result.score == 0.0
        assert "error" in result.reasoning

    def test_empty_string_output(self):
        item = EvalInputItem(
            id="empty",
            input_obj="{}",
            expected_output_obj="{}",
            output_obj="",
            full_dataset_entry={},
        )
        result = _evaluate_single(item, _DB_DIR)
        assert result.score == 0.0
