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

import pytest
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example
from langsmith.schemas import Run
from pydantic import ValidationError

from nat.builder.evaluator import EvaluatorInfo
from nat.data_models.evaluator import EvalInput
from nat.data_models.evaluator import EvalOutputItem
from nat.plugins.langchain.eval.langsmith_evaluator import LangSmithEvaluatorConfig
from nat.plugins.langchain.eval.langsmith_evaluator import register_langsmith_evaluator
from nat.plugins.langchain.eval.langsmith_evaluator_adapter import LangSmithEvaluatorAdapter

from .conftest import make_mock_builder
from .conftest import register_evaluator_ctx


async def _register(config, builder=None):
    """Drive the async context manager returned by register_langsmith_evaluator."""
    return await register_evaluator_ctx(register_langsmith_evaluator, config, builder)


# --------------------------------------------------------------------------- #
# Config validation (registry-based)
# --------------------------------------------------------------------------- #


class TestConfigValidation:
    """Tests for LangSmithEvaluatorConfig validation with registry lookup."""

    def test_valid_evaluator_name(self):
        """Config with a known evaluator name is valid."""
        config = LangSmithEvaluatorConfig(evaluator="exact_match")
        assert config.evaluator == "exact_match"

    def test_unknown_evaluator_raises(self):
        """Config with an unknown evaluator name raises ValueError."""
        with pytest.raises(ValidationError, match="Unknown evaluator"):
            LangSmithEvaluatorConfig(evaluator="nonexistent_evaluator")

    def test_evaluator_required(self):
        """Omitting 'evaluator' raises a validation error."""
        with pytest.raises(ValidationError):
            LangSmithEvaluatorConfig()

    def test_error_message_lists_available(self):
        """Error message includes available evaluator names."""
        with pytest.raises(ValidationError, match="exact_match"):
            LangSmithEvaluatorConfig(evaluator="bogus")

    def test_error_message_suggests_custom(self):
        """Error message suggests langsmith_custom for dotted paths."""
        with pytest.raises(ValidationError, match="langsmith_custom"):
            LangSmithEvaluatorConfig(evaluator="my_package.my_evaluator")


# --------------------------------------------------------------------------- #
# Registration through registry
# --------------------------------------------------------------------------- #


class TestRegistryEvaluatorRegistration:
    """Tests driven through register_langsmith_evaluator with registry names."""

    async def test_exact_match(self, eval_input_matching, eval_input_non_matching):
        """exact_match registered and evaluated by short name."""
        config = LangSmithEvaluatorConfig(evaluator="exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)

        assert isinstance(info, EvaluatorInfo)
        assert "exact_match" in info.description

        output_match = await info.evaluate_fn(eval_input_matching)
        assert output_match.eval_output_items[0].score is True

        output_mismatch = await info.evaluate_fn(eval_input_non_matching)
        assert output_mismatch.eval_output_items[0].score is False

    async def test_multi_item(self, eval_input_multi_item):
        """Evaluator processes multiple items correctly through registration."""
        config = LangSmithEvaluatorConfig(evaluator="exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)
        output = await info.evaluate_fn(eval_input_multi_item)

        assert len(output.eval_output_items) == 3
        scores_by_id = {item.id: item.score for item in output.eval_output_items}
        assert scores_by_id["multi_1"] is True  # Paris == Paris
        assert scores_by_id["multi_2"] is False  # Berlin != Munich
        assert scores_by_id["multi_3"] is True  # Tokyo == Tokyo

    async def test_empty_input(self):
        """Evaluator handles empty input gracefully through registration."""
        config = LangSmithEvaluatorConfig(evaluator="exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)
        output = await info.evaluate_fn(EvalInput(eval_input_items=[]))

        assert output.eval_output_items == []
        assert output.average_score is None

    async def test_evaluator_info_metadata(self):
        """EvaluatorInfo returned by registration has correct config and description."""
        config = LangSmithEvaluatorConfig(evaluator="exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)

        assert info.config is config
        assert "exact_match" in info.description


# --------------------------------------------------------------------------- #
# LangSmithEvaluatorAdapter (direct instantiation tests)
#
# These test the adapter directly, not through the plugin config.
# All conventions remain valid (used by langsmith_custom).
# --------------------------------------------------------------------------- #


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

    Covers evaluator conventions that cannot be referenced by a registry
    name: RunEvaluator subclasses, (run, example) functions, and custom
    openevals-style functions defined inline.

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

        def custom_scorer(*, inputs=None, outputs=None, reference_outputs=None):  # noqa: ARG001
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

    async def test_adapter_passes_extra_fields(self, eval_input_with_context):
        """LangSmithEvaluatorAdapter passes extra_fields through to evaluator."""
        received_kwargs = {}

        def capture_evaluator(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):
            received_kwargs.update(kwargs)
            received_kwargs["inputs"] = inputs
            received_kwargs["outputs"] = outputs
            return {"key": "test", "score": True}

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=capture_evaluator,
            convention="openevals_function",
            max_concurrency=1,
            extra_fields={"context": "retrieved_context"},
        )
        await evaluator.evaluate(eval_input_with_context)

        assert received_kwargs["context"] == "Doodads are small mechanical gadgets used in workshops."


# --------------------------------------------------------------------------- #
# LangSmithEvaluatorConfig extra_fields (registry-based)
# --------------------------------------------------------------------------- #


class TestLangSmithEvaluatorConfigExtraFields:
    """Tests for extra_fields on the registry-based langsmith evaluator config."""

    def test_extra_fields_default_none(self):
        config = LangSmithEvaluatorConfig(evaluator="exact_match")
        assert config.extra_fields is None

    def test_extra_fields_accepted(self):
        config = LangSmithEvaluatorConfig(
            evaluator="exact_match",
            extra_fields={"context": "ctx_field"},
        )
        assert config.extra_fields == {"context": "ctx_field"}
