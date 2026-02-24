# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Unit tests for OTEL run matching helpers and _match_and_link_otel_runs."""

from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _backfill_feedback_for_unlinked_items
from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _eager_link_run_to_item
from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _get_run_input_str
from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _link_run_to_item
from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _match_and_link_otel_runs
from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _retry_unlinked_references
from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _span_id_to_langsmith_run_id

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_run(run_id: str, input_text: str, name: str = "<workflow>") -> MagicMock:
    run = MagicMock()
    run.id = run_id
    run.name = name
    run.inputs = {"input": input_text}
    run.reference_example_id = None
    return run


def _mock_item(
    item_id: str,
    input_text: str,
    scores: dict | None = None,
    reasoning: dict | None = None,
    root_span_id: int | None = None,
) -> MagicMock:
    item = MagicMock()
    item.item_id = item_id
    item.input_obj = input_text
    item.scores = scores if scores is not None else {"accuracy": 0.9}
    item.reasoning = reasoning if reasoning is not None else {"accuracy": "correct"}
    item.root_span_id = root_span_id
    return item


def _mock_eval_result(items: list[MagicMock]) -> MagicMock:
    result = MagicMock()
    result.items = items
    return result


# ===========================================================================
# _get_run_input_str
# ===========================================================================


class TestGetRunInputStr:

    def test_dict_with_input_key(self):
        run = MagicMock()
        run.inputs = {"input": "What is X?"}
        assert _get_run_input_str(run) == "What is X?"

    def test_dict_without_input_key(self):
        run = MagicMock()
        run.inputs = {"prompt": "What is X?"}
        assert _get_run_input_str(run) == ""

    def test_plain_string(self):
        run = MagicMock()
        run.inputs = "What is X?"
        assert _get_run_input_str(run) == "What is X?"

    def test_none_input(self):
        run = MagicMock()
        run.inputs = None
        assert _get_run_input_str(run) == ""

    def test_nested_dict(self):
        run = MagicMock()
        run.inputs = {"input": {"text": "Q"}}
        assert _get_run_input_str(run) == "{'text': 'Q'}"


# ===========================================================================
# _link_run_to_item
# ===========================================================================


class TestLinkRunToItem:

    def test_happy_path(self):
        client = MagicMock()
        run = _mock_run("r1", "q1")
        item = _mock_item("i1", "q1", scores={"acc": 0.9}, reasoning={"acc": "ok"})

        result = _link_run_to_item(client, run, item, {"i1": "ex-1"})
        assert result is True
        client.update_run.assert_called_once_with("r1", reference_example_id="ex-1")
        client.create_feedback.assert_called_once()

    def test_missing_example_id(self):
        client = MagicMock()
        result = _link_run_to_item(client, _mock_run("r1", "q1"), _mock_item("i1", "q1"), {})
        assert result is False
        client.update_run.assert_not_called()

    def test_update_run_fails(self):
        client = MagicMock()
        client.update_run.side_effect = Exception("API error")
        result = _link_run_to_item(client, _mock_run("r1", "q1"), _mock_item("i1", "q1"), {"i1": "ex-1"})
        assert result is False

    def test_create_feedback_fails_partially(self):
        client = MagicMock()
        client.create_feedback.side_effect = [None, Exception("fail"), None]
        item = _mock_item("i1", "q1", scores={"a": 0.9, "b": 0.8, "c": 0.7}, reasoning={"a": "", "b": "", "c": ""})
        result = _link_run_to_item(client, _mock_run("r1", "q1"), item, {"i1": "ex-1"})
        # update_run succeeded; partial feedback failure is tolerated
        assert result is True
        assert client.create_feedback.call_count == 3

    def test_empty_scores(self):
        client = MagicMock()
        item = _mock_item("i1", "q1", scores={}, reasoning={})
        result = _link_run_to_item(client, _mock_run("r1", "q1"), item, {"i1": "ex-1"})
        assert result is True
        client.create_feedback.assert_not_called()


# ===========================================================================
# _match_and_link_otel_runs — substring matching
# ===========================================================================


class TestMatchSubstring:

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_all_exact_matches(self, _sleep):
        client = MagicMock()
        runs = [_mock_run(f"r{i}", f"question-{i:04d}") for i in range(5)]
        items = [_mock_item(f"i{i}", f"question-{i:04d}") for i in range(5)]
        client.list_runs.return_value = runs

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result(items),
                                            example_ids={f"i{i}": f"ex-{i}"
                                                         for i in range(5)},
                                            expected_count=5,
                                            max_retries=1,
                                            retry_delay=0)

        assert matched == 5

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_item_substring_of_run(self, _sleep):
        """Item text is a substring of the OTEL run's input."""
        client = MagicMock()
        client.list_runs.return_value = [_mock_run("r1", "Q: What is X?")]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([_mock_item("i1", "What is X?")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=1,
                                            retry_delay=0)

        assert matched == 1

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_run_substring_of_item(self, _sleep):
        """OTEL run's input is a substring of the item text."""
        client = MagicMock()
        client.list_runs.return_value = [_mock_run("r1", "What is X?")]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result(
                                                [_mock_item("i1", "Query: What is X? Please answer.")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=1,
                                            retry_delay=0)

        assert matched == 1

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_no_matches(self, _sleep):
        """Completely different inputs — nothing matches."""
        client = MagicMock()
        client.list_runs.return_value = [_mock_run("r1", "abc")]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([_mock_item("i1", "xyz")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=1,
                                            retry_delay=0)

        assert matched == 0

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_duplicate_inputs_match_different_items(self, _sleep):
        """Two runs with identical input should each match a different item."""
        client = MagicMock()
        client.list_runs.return_value = [
            _mock_run("r1", "same question"),
            _mock_run("r2", "same question"),
        ]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([
                                                _mock_item("i1", "same question"),
                                                _mock_item("i2", "same question"),
                                            ]),
                                            example_ids={
                                                "i1": "ex-1", "i2": "ex-2"
                                            },
                                            expected_count=2,
                                            max_retries=1,
                                            retry_delay=0)

        # Both should match — r1→i1, r2→i2 (i1 removed from pool after first match)
        assert matched == 2

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_100_items_no_substring_collision(self, _sleep):
        """100 zero-padded items all match exactly."""
        client = MagicMock()
        n = 100
        runs = [_mock_run(f"r{i}", f"q-{i:04d}") for i in range(n)]
        items = [_mock_item(f"i{i}", f"q-{i:04d}") for i in range(n)]
        client.list_runs.return_value = runs

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result(items),
                                            example_ids={f"i{i}": f"ex-{i}"
                                                         for i in range(n)},
                                            expected_count=n,
                                            max_retries=1,
                                            retry_delay=0)

        assert matched == n


# ===========================================================================
# _match_and_link_otel_runs — retry logic
# ===========================================================================


class TestMatchRetryLogic:

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_all_runs_first_attempt(self, _sleep):
        client = MagicMock()
        client.list_runs.return_value = [_mock_run("r1", "q1")]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([_mock_item("i1", "q1")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=3,
                                            retry_delay=0)

        assert matched == 1
        assert client.list_runs.call_count == 1

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_runs_arrive_incrementally(self, _sleep):
        client = MagicMock()
        batch1 = [_mock_run(f"r{i}", f"q-{i:04d}") for i in range(3)]
        batch2 = [_mock_run(f"r{i}", f"q-{i:04d}") for i in range(5)]
        client.list_runs.side_effect = [batch1, batch2]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result(
                                                [_mock_item(f"i{i}", f"q-{i:04d}") for i in range(5)]),
                                            example_ids={f"i{i}": f"ex-{i}"
                                                         for i in range(5)},
                                            expected_count=5,
                                            max_retries=3,
                                            retry_delay=0)

        assert matched == 5
        assert client.list_runs.call_count == 2

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_no_runs_ever(self, _sleep):
        client = MagicMock()
        client.list_runs.return_value = []

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([_mock_item("i1", "q1")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=3,
                                            retry_delay=0)

        assert matched == 0
        assert client.list_runs.call_count == 3

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_api_error_recovery(self, _sleep):
        client = MagicMock()
        client.list_runs.side_effect = [
            Exception("API error"),
            [_mock_run("r1", "q1")],
        ]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([_mock_item("i1", "q1")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=3,
                                            retry_delay=0)

        assert matched == 1

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_processed_run_ids_deduplication(self, _sleep):
        """Same run appearing in multiple attempts should only be processed once."""
        client = MagicMock()
        run = _mock_run("r1", "q1")
        client.list_runs.side_effect = [[run], [run]]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result([_mock_item("i1", "q1")]),
                                            example_ids={"i1": "ex-1"},
                                            expected_count=1,
                                            max_retries=2,
                                            retry_delay=0)

        assert matched == 1
        assert client.update_run.call_count == 1

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_missing_run_graceful(self, _sleep):
        """99 runs for 100 items — 1 OTEL trace was dropped."""
        client = MagicMock()
        client.list_runs.return_value = [_mock_run(f"r{i}", f"q-{i:04d}") for i in range(99)]

        matched = _match_and_link_otel_runs(client=client,
                                            project_name="test",
                                            eval_result=_mock_eval_result(
                                                [_mock_item(f"i{i}", f"q-{i:04d}") for i in range(100)]),
                                            example_ids={f"i{i}": f"ex-{i}"
                                                         for i in range(100)},
                                            expected_count=100,
                                            max_retries=2,
                                            retry_delay=0)

        assert matched == 99


# ===========================================================================
# _backfill_feedback_for_unlinked_items
# ===========================================================================


class TestBackfillFeedbackForUnlinkedItems:

    def test_no_backfill_when_linked_slots_cover_all_candidates(self):
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = "ex-1"
        client.list_runs.return_value = [run]

        item = _mock_item("i1",
                          "q1",
                          scores={"accuracy": 1.0},
                          reasoning={"accuracy": "ok"},
                          root_span_id=0x0123456789abcdef)
        count = _backfill_feedback_for_unlinked_items(client=client,
                                                      project_name="test",
                                                      items=[item],
                                                      example_ids={"i1": "ex-1"})

        assert count == 0
        client.create_feedback.assert_not_called()

    def test_backfills_run_feedback_for_unlinked_items(self):
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = None
        client.list_runs.return_value = [run]

        span_id = 0x0123456789abcdef
        expected_run_id = "00000000-0000-0000-0123-456789abcdef"
        item = _mock_item("i1",
                          "q1",
                          scores={
                              "accuracy": 0.5, "latency": 1.2
                          },
                          reasoning={
                              "accuracy": "wrong", "latency": "slow"
                          },
                          root_span_id=span_id)
        count = _backfill_feedback_for_unlinked_items(client=client,
                                                      project_name="test",
                                                      items=[item],
                                                      example_ids={"i1": "ex-1"})

        assert count == 1
        # Backfill only creates feedback — reference retry is handled by _retry_unlinked_references
        client.update_run.assert_not_called()
        assert client.create_feedback.call_count == 2
        keys = {call.kwargs["key"] for call in client.create_feedback.call_args_list}
        assert keys == {"accuracy", "latency"}
        assert all(call.kwargs["run_id"] == expected_run_id for call in client.create_feedback.call_args_list)

    def test_backfills_when_run_query_fails(self):
        client = MagicMock()
        client.list_runs.side_effect = Exception("api down")

        span_id = 0xf09206746ce2ad16
        expected_run_id = "00000000-0000-0000-f092-06746ce2ad16"
        item = _mock_item("i1", "q1", scores={"accuracy": 0.5}, reasoning={"accuracy": "wrong"}, root_span_id=span_id)
        count = _backfill_feedback_for_unlinked_items(client=client,
                                                      project_name="test",
                                                      items=[item],
                                                      example_ids={"i1": "ex-1"})

        assert count == 1
        # Backfill only creates feedback — no update_run retry
        client.update_run.assert_not_called()
        client.create_feedback.assert_called_once()
        assert client.create_feedback.call_args.kwargs["run_id"] == expected_run_id

    def test_skips_items_without_root_span_id(self):
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = None
        client.list_runs.return_value = [run]

        item = _mock_item("i1", "q1", scores={"accuracy": 0.5}, reasoning={"accuracy": "wrong"})
        count = _backfill_feedback_for_unlinked_items(client=client,
                                                      project_name="test",
                                                      items=[item],
                                                      example_ids={"i1": "ex-1"})

        assert count == 0
        client.create_feedback.assert_not_called()


# ===========================================================================
# _span_id_to_langsmith_run_id
# ===========================================================================


class TestSpanIdToRunId:

    def test_known_mapping(self):
        """Verify the deterministic span_id -> run_id formula."""
        span_id = 0x0123456789abcdef
        run_id = _span_id_to_langsmith_run_id(span_id)
        assert run_id == "00000000-0000-0000-0123-456789abcdef"

    def test_small_span_id_zero_padded(self):
        """Small span_ids should be zero-padded to 16 hex chars."""
        span_id = 0x1
        run_id = _span_id_to_langsmith_run_id(span_id)
        assert run_id == "00000000-0000-0000-0000-000000000001"

    def test_max_span_id(self):
        """64-bit max value."""
        span_id = 0xFFFFFFFFFFFFFFFF
        run_id = _span_id_to_langsmith_run_id(span_id)
        assert run_id == "00000000-0000-0000-ffff-ffffffffffff"

    def test_realistic_span_id(self):
        """A realistic span_id like those generated by _generate_nonzero_span_id."""
        span_id = 0xf09206746ce2ad16
        run_id = _span_id_to_langsmith_run_id(span_id)
        assert run_id == "00000000-0000-0000-f092-06746ce2ad16"


# ===========================================================================
# _eager_link_run_to_item
# ===========================================================================


class TestEagerLinkRunToItem:

    def test_happy_path(self):
        client = MagicMock()
        item = _mock_item("i1", "q1", scores={"acc": 0.9}, reasoning={"acc": "ok"})

        result = _eager_link_run_to_item(client, "00000000-0000-0000-0123-456789abcdef", item, {"i1": "ex-1"})
        assert result is True
        client.update_run.assert_called_once_with("00000000-0000-0000-0123-456789abcdef", reference_example_id="ex-1")
        client.create_feedback.assert_called_once()

    def test_missing_example_id(self):
        client = MagicMock()
        item = _mock_item("i1", "q1")
        result = _eager_link_run_to_item(client, "run-uuid", item, {})
        assert result is False
        client.update_run.assert_not_called()

    def test_update_run_fails_gracefully(self):
        client = MagicMock()
        client.update_run.side_effect = Exception("not found")
        item = _mock_item("i1", "q1")
        result = _eager_link_run_to_item(client, "run-uuid", item, {"i1": "ex-1"})
        assert result is False

    def test_feedback_failure_still_returns_true(self):
        """update_run succeeds but feedback fails — still considered linked."""
        client = MagicMock()
        client.create_feedback.side_effect = Exception("fail")
        item = _mock_item("i1", "q1", scores={"acc": 0.9}, reasoning={"acc": "ok"})
        result = _eager_link_run_to_item(client, "run-uuid", item, {"i1": "ex-1"})
        assert result is True

    def test_multiple_scores(self):
        """All feedback scores are attached."""
        client = MagicMock()
        item = _mock_item("i1", "q1", scores={"a": 0.9, "b": 0.8}, reasoning={"a": "good", "b": "fair"})
        _eager_link_run_to_item(client, "run-uuid", item, {"i1": "ex-1"})
        assert client.create_feedback.call_count == 2


# ===========================================================================
# _retry_unlinked_references
# ===========================================================================


class TestRetryUnlinkedReferences:

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_retries_unlinked_items(self, _sleep):
        """Items whose reference_example_id was silently dropped get retried."""
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = None
        # After retry, the run appears linked on the second check
        run_linked = _mock_run("r1", "q1")
        run_linked.reference_example_id = "ex-1"
        client.list_runs.side_effect = [[run], [run_linked]]

        span_id = 0x0123456789abcdef
        expected_run_id = "00000000-0000-0000-0123-456789abcdef"
        item = _mock_item("i1", "q1", root_span_id=span_id)
        retried = _retry_unlinked_references(
            client=client,
            project_name="test",
            items=[item],
            example_ids={"i1": "ex-1"},
            max_attempts=2,
            retry_delay=0,
        )

        assert retried == 1
        client.update_run.assert_called_once_with(expected_run_id, reference_example_id="ex-1")
        # No feedback is created — that's _backfill_feedback_for_unlinked_items' job
        client.create_feedback.assert_not_called()

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_skips_already_linked_items(self, _sleep):
        """Items whose reference_example_id is set don't get retried."""
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = "ex-1"
        client.list_runs.return_value = [run]

        item = _mock_item("i1", "q1", root_span_id=0x0123456789abcdef)
        retried = _retry_unlinked_references(
            client=client,
            project_name="test",
            items=[item],
            example_ids={"i1": "ex-1"},
            max_attempts=1,
            retry_delay=0,
        )

        assert retried == 0
        client.update_run.assert_not_called()

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_skips_items_without_span_id(self, _sleep):
        """Items without root_span_id are skipped."""
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = None
        client.list_runs.return_value = [run]

        item = _mock_item("i1", "q1")  # no root_span_id
        retried = _retry_unlinked_references(
            client=client,
            project_name="test",
            items=[item],
            example_ids={"i1": "ex-1"},
            max_attempts=1,
            retry_delay=0,
        )

        assert retried == 0
        client.update_run.assert_not_called()

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_handles_update_run_failure(self, _sleep):
        """update_run failure for one item doesn't block others."""
        client = MagicMock()
        run1 = _mock_run("r1", "q1")
        run1.reference_example_id = None
        run2 = _mock_run("r2", "q2")
        run2.reference_example_id = None
        # After first attempt: run1 still unlinked, run2 linked
        run2_linked = _mock_run("r2", "q2")
        run2_linked.reference_example_id = "ex-2"
        # After second attempt: run1 now linked too
        run1_linked = _mock_run("r1", "q1")
        run1_linked.reference_example_id = "ex-1"
        client.list_runs.side_effect = [
            [run1, run2],  # attempt 1: both unlinked
            [run1, run2_linked],  # attempt 2: run1 still unlinked
            [run1_linked, run2_linked],  # (not reached — only 2 attempts)
        ]
        # attempt 1: i1 fails, i2 succeeds; attempt 2: i1 succeeds
        client.update_run.side_effect = [Exception("fail"), None, None]

        items = [
            _mock_item("i1", "q1", root_span_id=0x1111111111111111),
            _mock_item("i2", "q2", root_span_id=0x2222222222222222),
        ]
        retried = _retry_unlinked_references(
            client=client,
            project_name="test",
            items=items,
            example_ids={
                "i1": "ex-1", "i2": "ex-2"
            },
            max_attempts=2,
            retry_delay=0,
        )

        assert retried == 2  # i2 on attempt 1, i1 on attempt 2
        assert client.update_run.call_count == 3

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_returns_zero_when_all_linked(self, _sleep):
        """All items already linked — nothing to retry."""
        client = MagicMock()
        run1 = _mock_run("r1", "q1")
        run1.reference_example_id = "ex-1"
        run2 = _mock_run("r2", "q2")
        run2.reference_example_id = "ex-2"
        client.list_runs.return_value = [run1, run2]

        items = [
            _mock_item("i1", "q1", root_span_id=0x1111111111111111),
            _mock_item("i2", "q2", root_span_id=0x2222222222222222),
        ]
        retried = _retry_unlinked_references(
            client=client,
            project_name="test",
            items=items,
            example_ids={
                "i1": "ex-1", "i2": "ex-2"
            },
            max_attempts=3,
            retry_delay=0,
        )

        assert retried == 0
        client.update_run.assert_not_called()

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_retries_multiple_attempts_until_linked(self, _sleep):
        """Retries across multiple attempts until all items are linked."""
        client = MagicMock()
        run = _mock_run("r1", "q1")
        run.reference_example_id = None
        run_still_unlinked = _mock_run("r1", "q1")
        run_still_unlinked.reference_example_id = None
        run_linked = _mock_run("r1", "q1")
        run_linked.reference_example_id = "ex-1"
        # Attempt 1: unlinked → retry. Attempt 2: still unlinked → retry again. Attempt 3: linked.
        client.list_runs.side_effect = [[run], [run_still_unlinked], [run_linked]]

        item = _mock_item("i1", "q1", root_span_id=0x0123456789abcdef)
        retried = _retry_unlinked_references(
            client=client,
            project_name="test",
            items=[item],
            example_ids={"i1": "ex-1"},
            max_attempts=3,
            retry_delay=0,
        )

        assert retried == 2  # retried on attempt 1 and 2, stopped at attempt 3
        assert client.update_run.call_count == 2
        assert _sleep.call_count == 2  # slept between attempts 1→2 and 2→3
