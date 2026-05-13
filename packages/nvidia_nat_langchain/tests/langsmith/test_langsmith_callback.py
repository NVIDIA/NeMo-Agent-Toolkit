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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.eval.eval_callbacks import EvalResult
from nat.eval.eval_callbacks import EvalResultItem
from nat.eval.evaluator.evaluator_model import EvalInputItem
from nat.profiler.parameter_optimization.optimizer_callbacks import TrialResult


class TestLangSmithEvaluationCallback:

    @pytest.fixture(autouse=True)
    def mock_langsmith(self):
        with patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.langsmith") as mock_ls:
            mock_client = MagicMock()
            mock_ls.Client.return_value = mock_client
            mock_ls.utils.LangSmithConflictError = type("LangSmithConflictError", (Exception, ), {})
            self.mock_client = mock_client
            yield

    @pytest.fixture
    def eval_cb(self):
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import LangSmithEvaluationCallback
        return LangSmithEvaluationCallback(project="test-proj")

    def test_on_dataset_loaded_creates_dataset(self, eval_cb):
        mock_ds = MagicMock()
        mock_ds.id = "ds-1"
        self.mock_client.create_dataset.return_value = mock_ds
        eval_cb.on_dataset_loaded(
            dataset_name="ds",
            items=[EvalInputItem(id="q1", input_obj="q", expected_output_obj="a", full_dataset_entry={})])
        self.mock_client.create_dataset.assert_called_once()
        self.mock_client.create_example.assert_called_once()

    def test_on_dataset_loaded_stores_example_ids(self, eval_cb):
        mock_ds = MagicMock()
        mock_ds.id = "ds-1"
        self.mock_client.create_dataset.return_value = mock_ds
        mock_example = MagicMock()
        mock_example.id = "ex-1"
        self.mock_client.create_example.return_value = mock_example
        eval_cb.on_dataset_loaded(
            dataset_name="ds",
            items=[EvalInputItem(id="q1", input_obj="q", expected_output_obj="a", full_dataset_entry={})])
        assert eval_cb._example_ids["q1"] == "ex-1"

    def test_on_dataset_loaded_reuses_existing_dataset_and_loads_examples(self, eval_cb):
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import langsmith
        self.mock_client.create_dataset.side_effect = langsmith.utils.LangSmithConflictError("exists")
        mock_existing = MagicMock()
        mock_existing.id = "ds-existing"
        self.mock_client.read_dataset.return_value = mock_existing
        # Mock list_examples to return existing examples with nat_item_id
        mock_ex = MagicMock()
        mock_ex.id = "ex-existing"
        mock_ex.inputs = {"nat_item_id": "1", "question": "q"}
        self.mock_client.list_examples.return_value = [mock_ex]
        eval_cb.on_dataset_loaded(
            dataset_name="existing",
            items=[EvalInputItem(id=1, input_obj="q", expected_output_obj="a", full_dataset_entry={})])
        self.mock_client.read_dataset.assert_called_once_with(dataset_name="Benchmark Dataset (Existing)")
        self.mock_client.create_example.assert_not_called()
        # Should have loaded the existing example ID keyed by nat_item_id
        assert eval_cb._example_ids["1"] == "ex-existing"


class TestMatchAndLinkOtelRuns:
    """Tests for _match_and_link_otel_runs and _normalize_input."""

    def test_normalize_strips_json_quotes(self):
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _normalize_input
        assert _normalize_input('"What is 2+2?"') == "What is 2+2?"

    def test_normalize_preserves_plain_text(self):
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _normalize_input
        assert _normalize_input("What is 2+2?") == "What is 2+2?"

    def test_normalize_strips_whitespace(self):
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _normalize_input
        assert _normalize_input("  hello  ") == "hello"

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_matched_items_not_rematched_across_retries(self, mock_sleep):
        """Items matched in attempt 1 should not steal runs in attempt 2."""
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _match_and_link_otel_runs

        mock_client = MagicMock()

        # Attempt 1: returns run for q1 only
        run_q1 = MagicMock()
        run_q1.id = "run-1"
        run_q1.name = "<workflow>"
        run_q1.inputs = {"input": "What is 2+2?"}
        run_q1.reference_example_id = None

        # Attempt 2: returns run for q2
        run_q2 = MagicMock()
        run_q2.id = "run-2"
        run_q2.name = "<workflow>"
        run_q2.inputs = {"input": "What is 3+3?"}
        run_q2.reference_example_id = None

        mock_client.list_runs.side_effect = [[run_q1], [run_q2]]
        mock_client.update_run.return_value = None
        mock_client.create_feedback.return_value = None

        eval_result = MagicMock()
        item_q1 = MagicMock()
        item_q1.item_id = "q1"
        item_q1.input_obj = "What is 2+2?"
        item_q1.scores = {"acc": 1.0}
        item_q1.reasoning = {}

        item_q2 = MagicMock()
        item_q2.item_id = "q2"
        item_q2.input_obj = "What is 3+3?"
        item_q2.scores = {"acc": 1.0}
        item_q2.reasoning = {}

        eval_result.items = [item_q1, item_q2]

        matched = _match_and_link_otel_runs(
            client=mock_client,
            project_name="test",
            eval_result=eval_result,
            example_ids={
                "q1": "ex-1", "q2": "ex-2"
            },
            expected_count=2,
            max_retries=2,
            retry_delay=0,
        )
        assert matched == 2

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_longest_substring_match_wins(self, mock_sleep):
        """When multiple items match a run, prefer the longest match."""
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _match_and_link_otel_runs

        mock_client = MagicMock()

        # Run input contains the longer question
        run = MagicMock()
        run.id = "run-1"
        run.name = "<workflow>"
        run.inputs = {"input": "Who is the president of France?"}
        run.reference_example_id = None

        mock_client.list_runs.return_value = [run]
        mock_client.update_run.return_value = None
        mock_client.create_feedback.return_value = None

        eval_result = MagicMock()

        # Short question (substring of the run input)
        item_short = MagicMock()
        item_short.item_id = "short"
        item_short.input_obj = "president"
        item_short.scores = {"acc": 1.0}
        item_short.reasoning = {}

        # Long question (exact match)
        item_long = MagicMock()
        item_long.item_id = "long"
        item_long.input_obj = "Who is the president of France?"
        item_long.scores = {"acc": 1.0}
        item_long.reasoning = {}

        eval_result.items = [item_short, item_long]

        _match_and_link_otel_runs(
            client=mock_client,
            project_name="test",
            eval_result=eval_result,
            example_ids={
                "short": "ex-1", "long": "ex-2"
            },
            expected_count=2,
            max_retries=1,
            retry_delay=0,
        )
        # The long question should match (exact), not the short one
        calls = mock_client.update_run.call_args_list
        linked_example_ids = [c.kwargs.get("reference_example_id") or c[1].get("reference_example_id") for c in calls]
        assert "ex-2" in linked_example_ids

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_json_quoted_input_matches(self, mock_sleep):
        """Runs with JSON-quoted inputs should still match plain-text items."""
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import _match_and_link_otel_runs

        mock_client = MagicMock()

        run = MagicMock()
        run.id = "run-1"
        run.name = "<workflow>"
        # OTEL serializes plain strings with JSON quoting
        run.inputs = {"input": '"What is quantum computing?"'}
        run.reference_example_id = None

        mock_client.list_runs.return_value = [run]
        mock_client.update_run.return_value = None
        mock_client.create_feedback.return_value = None

        eval_result = MagicMock()
        item = MagicMock()
        item.item_id = "q1"
        item.input_obj = "What is quantum computing?"
        item.scores = {"acc": 1.0}
        item.reasoning = {}
        eval_result.items = [item]

        matched = _match_and_link_otel_runs(
            client=mock_client,
            project_name="test",
            eval_result=eval_result,
            example_ids={"q1": "ex-1"},
            expected_count=1,
            max_retries=1,
            retry_delay=0,
        )
        assert matched == 1


class TestLangSmithEvaluationCallbackLinking:

    @pytest.fixture(autouse=True)
    def mock_langsmith(self):
        with patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.langsmith") as mock_ls:
            mock_client = MagicMock()
            mock_ls.Client.return_value = mock_client
            mock_ls.utils.LangSmithConflictError = type("LangSmithConflictError", (Exception, ), {})
            self.mock_client = mock_client
            yield

    @pytest.fixture
    def eval_cb(self):
        from nat.plugins.langchain.langsmith.langsmith_evaluation_callback import LangSmithEvaluationCallback
        return LangSmithEvaluationCallback(project="test-proj")

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_on_eval_complete_links_otel_runs(self, _mock_sleep, eval_cb):
        # Simulate dataset already loaded
        eval_cb._dataset_id = "ds-1"
        eval_cb._example_ids = {"q1": "ex-1", "q2": "ex-2"}

        # Mock OTEL runs
        mock_run1 = MagicMock()
        mock_run1.id = "otel-1"
        mock_run1.name = "<workflow>"
        mock_run1.inputs = {"input": "2+2"}
        mock_run2 = MagicMock()
        mock_run2.id = "otel-2"
        mock_run2.name = "<workflow>"
        mock_run2.inputs = {"input": "3*3"}
        self.mock_client.list_runs.return_value = [mock_run1, mock_run2]

        eval_cb.on_eval_complete(
            EvalResult(
                metric_scores={"accuracy": 0.9},
                items=[
                    EvalResultItem(item_id="q1",
                                   input_obj="2+2",
                                   expected_output="4",
                                   actual_output="4",
                                   scores={"accuracy": 1.0},
                                   reasoning={"accuracy": "correct"}),
                    EvalResultItem(item_id="q2",
                                   input_obj="3*3",
                                   expected_output="9",
                                   actual_output="8",
                                   scores={"accuracy": 0.8},
                                   reasoning={"accuracy": "wrong"}),
                ],
            ))
        # Should link OTEL runs to dataset examples (not create synthetic runs)
        assert self.mock_client.update_run.call_count == 2
        assert self.mock_client.create_feedback.call_count == 2
        self.mock_client.create_run.assert_not_called()

    def test_on_eval_complete_skips_without_dataset(self, eval_cb):
        # No dataset_id set — should skip gracefully
        eval_cb.on_eval_complete(EvalResult(metric_scores={"acc": 0.9}, items=[]))
        self.mock_client.update_run.assert_not_called()
        self.mock_client.create_run.assert_not_called()


class TestTemplateFormatDetection:
    """Tests for _detect_template_format, _validate_template_format, and _resolve_template_format."""

    @pytest.fixture
    def cb_cls(self):
        from nat.plugins.langchain.langsmith.langsmith_optimization_callback import LangSmithOptimizationCallback
        return LangSmithOptimizationCallback

    # ---- _detect_template_format ----

    @pytest.mark.parametrize(
        "text, expected",
        [
            # Jinja2 block tags
            ("Hello {% if x %}world{% endif %}", "jinja2"),
            ("{% for item in items %}{{ item }}{% endfor %}", "jinja2"),
            # Jinja2 comments
            ("Hello {# comment #} world", "jinja2"),
            # Jinja2 expression keywords inside {{ }}
            ("Hello {{ name | upper }}", "jinja2"),
            ("{{ x if y else z }}", "jinja2"),
            # Jinja2 plain variable (ambiguous with mustache, defaults jinja2)
            ("Hello {{ name }}", "jinja2"),
            # Mustache section markers
            ("{{#items}}{{name}}{{/items}}", "mustache"),
            ("{{>header}}", "mustache"),
            ("{{^empty}}fallback{{/empty}}", "mustache"),
            # F-string
            ("Hello {name}, welcome to {place}", "f-string"),
            ("Plain text no templates", "f-string"),
            ("", "f-string"),
        ],
    )
    def test_detect_template_format(self, cb_cls, text, expected):
        assert cb_cls._detect_template_format(text) == expected

    def test_detect_jinja2_block_takes_priority_over_mustache(self, cb_cls):
        # Mixed syntax: {% %} is unambiguous jinja2
        text = "{% if x %}{{#section}}content{{/section}}{% endif %}"
        assert cb_cls._detect_template_format(text) == "jinja2"

    def test_detect_mustache_not_confused_by_jinja2_comment_substring(self, cb_cls):
        # {{# contains {# as substring — mustache should win
        assert cb_cls._detect_template_format("{{#list}}item{{/list}}") == "mustache"

    # ---- _validate_template_format ----

    @pytest.mark.parametrize("fmt", ["f-string", "jinja2", "mustache"])
    def test_validate_accepts_valid_formats(self, cb_cls, fmt):
        assert cb_cls._validate_template_format(fmt) == fmt

    @pytest.mark.parametrize("fmt", ["invalid", "JINJA2", "fstring", ""])
    def test_validate_rejects_invalid_formats(self, cb_cls, fmt):
        with pytest.raises(ValueError, match="Invalid template_format"):
            cb_cls._validate_template_format(fmt)

    # ---- _resolve_template_format ----

    def test_resolve_uses_explicit_format_from_trial_result(self, cb_cls):
        cb = cb_cls.__new__(cb_cls)
        result = TrialResult(
            trial_number=0,
            parameters={},
            metric_scores={},
            is_best=False,
            prompts={"p": "Hello {{ name }}"},
            prompt_formats={"p": "mustache"},
        )
        # Explicit mustache should win over auto-detected jinja2
        assert cb._resolve_template_format("p", "Hello {{ name }}", result) == "mustache"

    def test_resolve_falls_back_to_auto_detection(self, cb_cls):
        cb = cb_cls.__new__(cb_cls)
        result = TrialResult(
            trial_number=0,
            parameters={},
            metric_scores={},
            is_best=False,
            prompts={"p": "Hello {% if x %}yes{% endif %}"},
        )
        assert cb._resolve_template_format("p", "Hello {% if x %}yes{% endif %}", result) == "jinja2"

    def test_resolve_falls_back_when_param_not_in_formats(self, cb_cls):
        cb = cb_cls.__new__(cb_cls)
        result = TrialResult(
            trial_number=0,
            parameters={},
            metric_scores={},
            is_best=False,
            prompts={"p": "Hello {name}"},
            prompt_formats={"other_param": "jinja2"},
        )
        assert cb._resolve_template_format("p", "Hello {name}", result) == "f-string"

    def test_resolve_validates_explicit_format(self, cb_cls):
        cb = cb_cls.__new__(cb_cls)
        result = TrialResult(
            trial_number=0,
            parameters={},
            metric_scores={},
            is_best=False,
            prompts={"p": "text"},
            prompt_formats={"p": "bad_format"},
        )
        with pytest.raises(ValueError, match="Invalid template_format"):
            cb._resolve_template_format("p", "text", result)


class TestLangSmithOptimizationCallback:

    @pytest.fixture(autouse=True)
    def mock_langsmith(self):
        with patch("nat.plugins.langchain.langsmith.langsmith_optimization_callback.langsmith") as mock_ls:
            mock_client = MagicMock()
            mock_ls.Client.return_value = mock_client
            mock_ls.utils.LangSmithConflictError = type("LangSmithConflictError", (Exception, ), {})
            self.mock_client = mock_client
            yield

    @pytest.fixture
    def opt_cb(self):
        from nat.plugins.langchain.langsmith.langsmith_optimization_callback import LangSmithOptimizationCallback
        return LangSmithOptimizationCallback(project="test-proj")

    @patch("nat.plugins.langchain.langsmith.langsmith_evaluation_callback.time.sleep")
    def test_on_trial_end_links_otel_runs(self, _mock_sleep, opt_cb):
        # Simulate dataset already created
        opt_cb._dataset_id = "ds-1"
        opt_cb._example_ids = {"q1": "ex-1"}
        opt_cb._run_number = 1

        # Mock project read for metadata update
        mock_project = MagicMock()
        mock_project.id = "proj-1"
        self.mock_client.read_project.return_value = mock_project

        # Mock OTEL runs returned by list_runs
        mock_otel_run = MagicMock()
        mock_otel_run.id = "otel-run-1"
        mock_otel_run.name = "<workflow>"
        mock_otel_run.inputs = {"input": "question-q1"}
        self.mock_client.list_runs.return_value = [mock_otel_run]

        eval_result = MagicMock()
        eval_item = MagicMock()
        eval_item.item_id = "q1"
        eval_item.input_obj = "question-q1"
        eval_item.expected_output = "answer"
        eval_item.scores = {"acc": 0.9}
        eval_item.reasoning = {"acc": "correct"}
        eval_result.items = [eval_item]

        opt_cb.on_trial_end(
            TrialResult(
                trial_number=0,
                parameters={"t": 0.7},
                metric_scores={"acc": 0.9},
                is_best=False,
                eval_result=eval_result,
            ))
        # Should link OTEL run to dataset example
        self.mock_client.update_run.assert_called_once_with("otel-run-1", reference_example_id="ex-1")
        # Should attach feedback
        self.mock_client.create_feedback.assert_called_once()
        # Should NOT create synthetic runs
        self.mock_client.create_run.assert_not_called()

    def test_on_trial_end_with_prompts(self, opt_cb):
        with patch(
                "nat.plugins.langchain.langsmith.langsmith_optimization_callback.LangSmithOptimizationCallback._push_prompt",
                return_value={"p": "url"},
        ) as mock_push:
            opt_cb.on_trial_end(
                TrialResult(
                    trial_number=0,
                    parameters={},
                    metric_scores={"acc": 0.9},
                    is_best=True,
                    prompts={"functions.agent.prompt": "You are helpful."},
                ))
            mock_push.assert_called_once()

    def test_on_study_end_flushes(self, opt_cb):
        best = TrialResult(trial_number=3, parameters={"t": 0.6}, metric_scores={"acc": 0.9}, is_best=True)
        opt_cb.on_study_end(best_trial=best, total_trials=10)
        self.mock_client.flush.assert_called_once()
        self.mock_client.create_run.assert_not_called()
