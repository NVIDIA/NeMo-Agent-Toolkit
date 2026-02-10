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

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example
from langsmith.schemas import Run

from nat.builder.evaluator import EvaluatorInfo
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.plugins.langchain.eval.langsmith_evaluator import LangSmithEvaluatorConfig
from nat.plugins.langchain.eval.langsmith_evaluator import register_langsmith_evaluator
from nat.plugins.langchain.eval.langsmith_evaluator_adapter import LangSmithEvaluatorAdapter


def _make_mock_builder(mock_llm=None):
    """Create a mock EvalBuilder with configurable get_llm."""
    builder = MagicMock(spec=["get_llm", "get_max_concurrency"])
    builder.get_llm = AsyncMock(return_value=mock_llm or MagicMock(name="mock_judge_llm"))
    builder.get_max_concurrency.return_value = 2
    return builder


async def _register(config, builder):
    """Drive the async context manager returned by register_langsmith_evaluator."""
    async with register_langsmith_evaluator(config, builder) as info:
        return info


class TestConfigValidation:
    """Tests for mutually exclusive mode validation on LangSmithEvaluatorConfig."""

    def test_evaluator_mode(self):
        """Config with only 'evaluator' is valid."""
        config = LangSmithEvaluatorConfig(evaluator="my_package.module.my_fn")
        assert config.evaluator == "my_package.module.my_fn"
        assert config.prompt is None
        assert config.llm_name is None

    def test_prompt_mode(self):
        """Config with 'prompt' + 'llm_name' is valid."""
        config = LangSmithEvaluatorConfig(prompt="correctness", llm_name="eval_llm")
        assert config.prompt == "correctness"
        assert config.llm_name == "eval_llm"
        assert config.evaluator is None

    def test_prompt_mode_defaults(self):
        """Prompt mode defaults are correct."""
        config = LangSmithEvaluatorConfig(prompt="correctness", llm_name="eval_llm")
        assert config.feedback_key == "score"
        assert config.continuous is False
        assert config.choices is None
        assert config.use_reasoning is True

    def test_prompt_mode_custom_options(self):
        """Prompt mode with custom scoring options."""
        config = LangSmithEvaluatorConfig(
            prompt="Is the output polite? {inputs} {outputs}",
            llm_name="eval_llm",
            feedback_key="politeness",
            continuous=True,
        )
        assert config.feedback_key == "politeness"
        assert config.continuous is True

    def test_neither_mode_raises(self):
        """Config with neither 'evaluator' nor 'prompt' raises."""
        with pytest.raises(ValueError, match="Exactly one of 'evaluator' or 'prompt'"):
            LangSmithEvaluatorConfig()

    def test_both_modes_raises(self):
        """Config with both 'evaluator' and 'prompt' raises."""
        with pytest.raises(ValueError, match="Exactly one of 'evaluator' or 'prompt'"):
            LangSmithEvaluatorConfig(
                evaluator="my_package.module.my_fn",
                prompt="correctness",
                llm_name="eval_llm",
            )

    def test_prompt_without_llm_name_raises(self):
        """Prompt mode requires llm_name."""
        with pytest.raises(ValueError, match="'llm_name' is required"):
            LangSmithEvaluatorConfig(prompt="correctness")

    def test_continuous_and_choices_raises(self):
        """continuous and choices are mutually exclusive."""
        with pytest.raises(ValueError, match="'continuous' and 'choices' are mutually exclusive"):
            LangSmithEvaluatorConfig(
                prompt="correctness",
                llm_name="eval_llm",
                continuous=True,
                choices=[0.0, 0.5, 1.0],
            )

    def test_choices_without_continuous_is_valid(self):
        """choices alone (without continuous) is valid."""
        config = LangSmithEvaluatorConfig(
            prompt="correctness",
            llm_name="eval_llm",
            choices=[0.0, 0.5, 1.0],
        )
        assert config.choices == [0.0, 0.5, 1.0]
        assert config.continuous is False

    def test_evaluator_mode_warns_on_prompt_fields(self, caplog):
        """Evaluator mode logs a warning when prompt-mode-only fields are set."""
        import logging

        with caplog.at_level(logging.WARNING):
            LangSmithEvaluatorConfig(
                evaluator="my_package.module.my_fn",
                feedback_key="custom_key",
                continuous=True,
            )

        assert "only used in prompt mode" in caplog.text
        assert "feedback_key" in caplog.text
        assert "continuous" in caplog.text

    def test_evaluator_mode_no_warning_with_defaults(self, caplog):
        """Evaluator mode does NOT warn when all prompt-mode fields are at defaults."""
        import logging

        with caplog.at_level(logging.WARNING):
            LangSmithEvaluatorConfig(evaluator="my_package.module.my_fn")

        assert "only used in prompt mode" not in caplog.text

    def test_evaluator_accepts_any_path(self):
        """Evaluator mode accepts any string; import errors happen at registration."""
        config = LangSmithEvaluatorConfig(evaluator="nonexistent.path")
        assert config.evaluator == "nonexistent.path"


class TestCustomEvaluatorRegistration:
    """Tests driven through register_langsmith_evaluator with a mock builder.

    Covers all scenarios where the evaluator is referenced by a real importable
    dotted path (prebuilt openevals functions) and error cases for bad paths.
    """

    async def test_openevals_exact_match(self, eval_input_matching, eval_input_non_matching):
        """openevals.exact_match registered and evaluated via dotted path."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = _make_mock_builder()

        info = await _register(config, builder)

        assert isinstance(info, EvaluatorInfo)
        assert "openevals.exact_match" in info.description

        output_match = await info.evaluate_fn(eval_input_matching)
        assert output_match.eval_output_items[0].score is True

        output_mismatch = await info.evaluate_fn(eval_input_non_matching)
        assert output_mismatch.eval_output_items[0].score is False

    async def test_openevals_exact_match_async(self, eval_input_matching, eval_input_non_matching):
        """openevals.exact_match_async registered and evaluated via dotted path."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match_async")
        builder = _make_mock_builder()

        info = await _register(config, builder)

        output_match = await info.evaluate_fn(eval_input_matching)
        assert output_match.eval_output_items[0].score is True

        output_mismatch = await info.evaluate_fn(eval_input_non_matching)
        assert output_mismatch.eval_output_items[0].score is False

    async def test_multi_item(self, eval_input_multi_item):
        """Evaluator processes multiple items correctly through registration."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = _make_mock_builder()

        info = await _register(config, builder)
        output = await info.evaluate_fn(eval_input_multi_item)

        assert len(output.eval_output_items) == 3
        scores_by_id = {item.id: item.score for item in output.eval_output_items}
        assert scores_by_id["multi_1"] is True  # Paris == Paris
        assert scores_by_id["multi_2"] is False  # Berlin != Munich
        assert scores_by_id["multi_3"] is True  # Tokyo == Tokyo

    async def test_empty_input(self):
        """Evaluator handles empty input gracefully through registration."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = _make_mock_builder()

        info = await _register(config, builder)
        output = await info.evaluate_fn(EvalInput(eval_input_items=[]))

        assert output.eval_output_items == []
        assert output.average_score is None

    async def test_evaluator_info_metadata(self):
        """EvaluatorInfo returned by registration has correct config and description."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = _make_mock_builder()

        info = await _register(config, builder)

        assert info.config is config
        assert "exact_match" in info.description

    async def test_nonexistent_module_raises(self):
        """Registration raises ImportError for a nonexistent module."""
        config = LangSmithEvaluatorConfig(evaluator="nonexistent_package.foo")
        builder = _make_mock_builder()

        with pytest.raises(ImportError, match="Could not import module"):
            await _register(config, builder)

    async def test_nonexistent_attribute_raises(self):
        """Registration raises AttributeError for a nonexistent attribute."""
        config = LangSmithEvaluatorConfig(evaluator="json.nonexistent_function")
        builder = _make_mock_builder()

        with pytest.raises(AttributeError, match="has no attribute"):
            await _register(config, builder)

    async def test_bad_path_format_raises(self):
        """Registration raises ValueError for a path without a dot."""
        config = LangSmithEvaluatorConfig(evaluator="no_dot_in_path")
        builder = _make_mock_builder()

        with pytest.raises(ValueError, match="Invalid evaluator path"):
            await _register(config, builder)

    async def test_class_requiring_args_raises(self):
        """Registration raises TypeError for classes needing constructor arguments."""
        config = LangSmithEvaluatorConfig(evaluator="datetime.datetime")
        builder = _make_mock_builder()

        with pytest.raises(TypeError, match="Could not instantiate class"):
            await _register(config, builder)


class SimpleRunEvaluator(RunEvaluator):
    """A minimal RunEvaluator that checks if run outputs match example outputs."""

    def evaluate_run(self,
                     run: Run,
                     example: Example | None = None,
                     evaluator_run_id=None,
                     **kwargs) -> EvaluationResult:
        if example is None:
            return EvaluationResult(key="simple", score=0.0, comment="No example provided")

        matches = run.outputs == example.outputs
        return EvaluationResult(
            key="simple",
            score=1.0 if matches else 0.0,
            comment="Match" if matches else "Mismatch",
        )


def _run_example_evaluator(run: Run, example: Example | None = None) -> EvaluationResult:
    """A simple function evaluator with (run, example) signature."""
    if example and run.outputs == example.outputs:
        return EvaluationResult(key="fn_eval", score=1.0)
    return EvaluationResult(key="fn_eval", score=0.0)


class TestLangSmithEvaluatorAdapter:
    """Tests for LangSmithEvaluatorAdapter with direct instantiation.

    Covers evaluator conventions that cannot be referenced by a real
    importable dotted path: RunEvaluator subclasses, (run, example)
    functions, and custom openevals-style functions defined inline.

    Follows the same direct-instantiation pattern used by other NAT
    evaluator tests (RAGEvaluator, TrajectoryEvaluator, etc.).
    Convention strings are used instead of the private _EvaluatorConvention enum.
    """

    async def test_run_evaluator_match(self, eval_input_matching):
        """RunEvaluator subclass evaluates correctly (match)."""
        evaluator = LangSmithEvaluatorAdapter(
            evaluator=SimpleRunEvaluator(),
            convention="run_evaluator_class",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert isinstance(item, EvalOutputItem)
        assert item.score == 1.0
        assert item.reasoning["comment"] == "Match"

    async def test_run_evaluator_mismatch(self, eval_input_non_matching):
        """RunEvaluator subclass evaluates correctly (mismatch)."""
        evaluator = LangSmithEvaluatorAdapter(
            evaluator=SimpleRunEvaluator(),
            convention="run_evaluator_class",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_non_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert item.score == 0.0
        assert item.reasoning["comment"] == "Mismatch"

    async def test_run_example_function_match(self, eval_input_matching):
        """Sync (run, example) function evaluates correctly (match)."""
        evaluator = LangSmithEvaluatorAdapter(
            evaluator=_run_example_evaluator,
            convention="run_example_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert item.score == 1.0
        assert item.reasoning["key"] == "fn_eval"

    async def test_run_example_function_mismatch(self, eval_input_non_matching):
        """Sync (run, example) function evaluates correctly (mismatch)."""
        evaluator = LangSmithEvaluatorAdapter(
            evaluator=_run_example_evaluator,
            convention="run_example_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_non_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert item.score == 0.0

    async def test_async_run_example_function(self, eval_input_matching):
        """Async (run, example) function is awaited properly."""

        async def async_re_eval(run, example=None):
            matches = run.outputs == (example.outputs if example else None)
            return EvaluationResult(key="async_fn", score=1.0 if matches else 0.0)

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=async_re_eval,
            convention="run_example_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        assert output.eval_output_items[0].score == 1.0

    async def test_custom_openevals_dict_with_metadata(self, eval_input_matching):
        """Custom function returning a dict with extra keys is handled."""

        def custom_scorer(*, inputs=None, outputs=None, reference_outputs=None):
            return {
                "key": "custom_key",
                "score": 0.75,
                "comment": "Partially correct",
            }

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=custom_scorer,
            convention="openevals_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert item.score == 0.75
        assert item.reasoning["comment"] == "Partially correct"

    async def test_custom_async_openevals_function(self, eval_input_matching):
        """Custom async function with openevals-style kwargs works."""

        async def async_eval(*, inputs=None, outputs=None, reference_outputs=None):
            match = outputs == reference_outputs
            return {"key": "custom_async", "score": match}

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=async_eval,
            convention="openevals_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        assert output.eval_output_items[0].score is True

    async def test_boolean_score_in_dict(self, eval_input_matching):
        """Custom function returning a dict with boolean score is handled."""

        def bool_scorer(*, inputs=None, outputs=None, reference_outputs=None):
            return {"key": "bool_check", "score": True}

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=bool_scorer,
            convention="openevals_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        assert output.eval_output_items[0].score is True

    async def test_evaluator_wraps_runtime_error(self, eval_input_matching):
        """RuntimeError in evaluator is wrapped into EvalOutputItem."""

        def bad_evaluator(*, inputs=None, outputs=None, reference_outputs=None):
            raise RuntimeError("Something broke")

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=bad_evaluator,
            convention="openevals_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert item.score == 0.0
        assert "Evaluator error" in item.reasoning["error"]
        assert "Something broke" in item.reasoning["error"]

    async def test_evaluator_wraps_value_error(self, eval_input_matching):
        """ValueError in evaluator is wrapped into EvalOutputItem."""

        def failing_evaluator(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            raise ValueError("Intentional test failure")

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=failing_evaluator,
            convention="openevals_function",
            max_concurrency=1,
        )
        output = await evaluator.evaluate(eval_input_matching)

        assert len(output.eval_output_items) == 1
        item = output.eval_output_items[0]
        assert item.score == 0.0
        assert "Evaluator error" in item.reasoning["error"]
        assert "Intentional test failure" in item.reasoning["error"]


class TestPromptModeRegistration:
    """Tests for the prompt/LLM-as-judge registration path using a mocked LLM.

    These tests exercise register_langsmith_evaluator end-to-end
    without requiring a real LLM by mocking builder.get_llm and
    create_llm_as_judge.
    """

    async def test_prompt_mode_builtin_prompt(self, eval_input_matching):
        """Prompt mode with a builtin prompt name creates a working evaluator."""
        mock_llm = MagicMock(name="mock_judge_llm")

        def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            return {"key": "score", "score": 0.9, "comment": "Looks good"}

        config = LangSmithEvaluatorConfig(prompt="correctness", llm_name="eval_llm")
        builder = _make_mock_builder(mock_llm)

        with patch(
                "openevals.llm.create_llm_as_judge",
                return_value=fake_judge,
        ) as mock_create:
            info = await _register(config, builder)

            # Verify create_llm_as_judge was called with the resolved prompt
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["judge"] is mock_llm
            assert call_kwargs["feedback_key"] == "score"
            assert call_kwargs["prompt"] != "correctness"  # resolved to full prompt text
            assert len(call_kwargs["prompt"]) > 50  # full openevals prompt is long

        # Verify the EvaluatorInfo is correct
        assert isinstance(info, EvaluatorInfo)
        assert "correctness" in info.description.lower()

        # Verify the evaluator works by calling evaluate_fn
        output = await info.evaluate_fn(eval_input_matching)
        assert len(output.eval_output_items) == 1
        assert output.eval_output_items[0].score == 0.9
        assert output.eval_output_items[0].reasoning["comment"] == "Looks good"

    async def test_prompt_mode_custom_prompt(self, eval_input_matching):
        """Prompt mode with a custom prompt template passes it through."""
        custom_prompt = "Rate professionalism: {inputs} {outputs}"
        mock_llm = MagicMock(name="mock_judge_llm")

        def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            return {"key": "professionalism", "score": 0.85, "comment": "Professional tone"}

        config = LangSmithEvaluatorConfig(
            prompt=custom_prompt,
            llm_name="eval_llm",
            feedback_key="professionalism",
            continuous=True,
        )
        builder = _make_mock_builder(mock_llm)

        with patch(
                "openevals.llm.create_llm_as_judge",
                return_value=fake_judge,
        ) as mock_create:
            info = await _register(config, builder)

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["prompt"] == custom_prompt
            assert call_kwargs["feedback_key"] == "professionalism"
            assert call_kwargs["continuous"] is True

        assert "custom LLM-as-judge" in info.description

        output = await info.evaluate_fn(eval_input_matching)
        assert output.eval_output_items[0].score == 0.85

    async def test_prompt_mode_with_choices(self, eval_input_matching):
        """Prompt mode passes choices through to create_llm_as_judge."""
        mock_llm = MagicMock(name="mock_judge_llm")

        def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            return {"key": "score", "score": 0.5}

        config = LangSmithEvaluatorConfig(
            prompt="correctness",
            llm_name="eval_llm",
            choices=[0.0, 0.5, 1.0],
        )
        builder = _make_mock_builder(mock_llm)

        with patch(
                "openevals.llm.create_llm_as_judge",
                return_value=fake_judge,
        ) as mock_create:
            info = await _register(config, builder)

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["choices"] == [0.0, 0.5, 1.0]
            assert call_kwargs["continuous"] is False

        output = await info.evaluate_fn(eval_input_matching)
        assert output.eval_output_items[0].score == 0.5

    async def test_prompt_mode_retry_applied(self):
        """When do_auto_retry is True, patch_with_retry is called on the LLM."""
        mock_llm = MagicMock(name="mock_judge_llm")
        patched_llm = MagicMock(name="patched_judge_llm")

        def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            return {"key": "score", "score": 1.0}

        config = LangSmithEvaluatorConfig(
            prompt="correctness",
            llm_name="eval_llm",
            do_auto_retry=True,
            num_retries=5,
        )
        builder = _make_mock_builder(mock_llm)

        with (
                patch(
                    "nat.plugins.langchain.eval.langsmith_evaluator.patch_with_retry",
                    return_value=patched_llm,
                ) as mock_retry,
                patch(
                    "openevals.llm.create_llm_as_judge",
                    return_value=fake_judge,
                ) as mock_create,
        ):
            await _register(config, builder)

            # Verify patch_with_retry was called with the original LLM
            mock_retry.assert_called_once()
            assert mock_retry.call_args[0][0] is mock_llm
            assert mock_retry.call_args[1]["retries"] == 5

            # Verify create_llm_as_judge received the patched LLM
            assert mock_create.call_args[1]["judge"] is patched_llm

    async def test_prompt_mode_retry_not_applied_when_disabled(self):
        """When do_auto_retry is explicitly False, patch_with_retry is NOT called."""
        mock_llm = MagicMock(name="mock_judge_llm")

        def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            return {"key": "score", "score": 1.0}

        config = LangSmithEvaluatorConfig(prompt="correctness", llm_name="eval_llm", do_auto_retry=False)
        builder = _make_mock_builder(mock_llm)

        with (
                patch("nat.plugins.langchain.eval.langsmith_evaluator.patch_with_retry", ) as mock_retry,
                patch(
                    "openevals.llm.create_llm_as_judge",
                    return_value=fake_judge,
                ),
        ):
            await _register(config, builder)
            mock_retry.assert_not_called()

    async def test_evaluator_mode_registration(self, eval_input_matching):
        """Evaluator mode (dotted path import) registration works end-to-end."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = _make_mock_builder()

        info = await _register(config, builder)

        assert isinstance(info, EvaluatorInfo)
        assert "openevals.exact_match" in info.description

        output = await info.evaluate_fn(eval_input_matching)
        assert len(output.eval_output_items) == 1
        assert output.eval_output_items[0].score is True

    async def test_evaluator_mode_registration_mismatch(self, eval_input_non_matching):
        """Evaluator mode registration produces correct results for mismatches."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = _make_mock_builder()

        info = await _register(config, builder)

        output = await info.evaluate_fn(eval_input_non_matching)
        assert output.eval_output_items[0].score is False
