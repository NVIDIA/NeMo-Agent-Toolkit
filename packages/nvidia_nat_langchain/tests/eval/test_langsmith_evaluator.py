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

from unittest.mock import patch

import pytest
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.evaluation.evaluator import RunEvaluator
from langsmith.schemas import Example
from langsmith.schemas import Run
from pydantic import ValidationError

from nat.builder.evaluator import EvaluatorInfo
from nat.eval.evaluator.evaluator_model import EvalInput
from nat.eval.evaluator.evaluator_model import EvalOutputItem
from nat.plugins.langchain.eval.langsmith_evaluator import LangSmithEvaluatorConfig
from nat.plugins.langchain.eval.langsmith_evaluator import register_langsmith_evaluator
from nat.plugins.langchain.eval.langsmith_evaluator_adapter import LangSmithEvaluatorAdapter

from .conftest import make_mock_builder
from .conftest import register_evaluator_ctx


async def _register(config, builder=None):
    """Drive the async context manager returned by register_langsmith_evaluator."""
    return await register_evaluator_ctx(register_langsmith_evaluator, config, builder)


class TestConfigValidation:
    """Tests for LangSmithEvaluatorConfig validation."""

    def test_evaluator_mode(self):
        """Config with 'evaluator' is valid."""
        config = LangSmithEvaluatorConfig(evaluator="my_package.module.my_fn")
        assert config.evaluator == "my_package.module.my_fn"

    def test_evaluator_accepts_any_path(self):
        """Evaluator accepts any string; import errors happen at registration."""
        config = LangSmithEvaluatorConfig(evaluator="nonexistent.path")
        assert config.evaluator == "nonexistent.path"

    def test_evaluator_required(self):
        """Omitting 'evaluator' raises a validation error."""
        with pytest.raises(ValidationError):
            LangSmithEvaluatorConfig()


class TestCustomEvaluatorRegistration:
    """Tests driven through register_langsmith_evaluator with a mock builder.

    Covers all scenarios where the evaluator is referenced by a real importable
    dotted path (prebuilt openevals functions) and error cases for bad paths.
    """

    async def test_openevals_exact_match(self, eval_input_matching, eval_input_non_matching):
        """openevals.exact_match registered and evaluated via dotted path."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = make_mock_builder()

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
        builder = make_mock_builder()

        info = await _register(config, builder)

        output_match = await info.evaluate_fn(eval_input_matching)
        assert output_match.eval_output_items[0].score is True

        output_mismatch = await info.evaluate_fn(eval_input_non_matching)
        assert output_mismatch.eval_output_items[0].score is False

    async def test_multi_item(self, eval_input_multi_item):
        """Evaluator processes multiple items correctly through registration."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
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
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)
        output = await info.evaluate_fn(EvalInput(eval_input_items=[]))

        assert output.eval_output_items == []
        assert output.average_score is None

    async def test_evaluator_info_metadata(self):
        """EvaluatorInfo returned by registration has correct config and description."""
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)

        assert info.config is config
        assert "exact_match" in info.description

    async def test_nonexistent_module_raises(self):
        """Registration raises ImportError for a nonexistent module."""
        config = LangSmithEvaluatorConfig(evaluator="nonexistent_package.foo")
        builder = make_mock_builder()

        with pytest.raises(ImportError, match="Could not import module"):
            await _register(config, builder)

    async def test_nonexistent_attribute_raises(self):
        """Registration raises AttributeError for a nonexistent attribute."""
        config = LangSmithEvaluatorConfig(evaluator="json.nonexistent_function")
        builder = make_mock_builder()

        with pytest.raises(AttributeError, match="has no attribute"):
            await _register(config, builder)

    async def test_bad_path_format_raises(self):
        """Registration raises ValueError for a path without a dot."""
        config = LangSmithEvaluatorConfig(evaluator="no_dot_in_path")
        builder = make_mock_builder()

        with pytest.raises(ValueError, match="Invalid evaluator path"):
            await _register(config, builder)

    async def test_class_requiring_args_raises(self):
        """Registration raises TypeError for classes needing constructor arguments."""
        config = LangSmithEvaluatorConfig(evaluator="datetime.datetime")
        builder = make_mock_builder()

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
# LangSmithEvaluatorConfig extra_fields
# --------------------------------------------------------------------------- #


class TestLangSmithEvaluatorConfigExtraFields:
    """Tests for extra_fields on the generic langsmith evaluator config."""

    def test_extra_fields_default_none(self):
        config = LangSmithEvaluatorConfig(evaluator="openevals.exact_match")
        assert config.extra_fields is None

    def test_extra_fields_accepted(self):
        config = LangSmithEvaluatorConfig(
            evaluator="openevals.exact_match",
            extra_fields={"context": "ctx_field"},
        )
        assert config.extra_fields == {"context": "ctx_field"}

    async def test_extra_fields_with_non_openevals_convention_warns_and_drops(self, caplog):
        """extra_fields warns and is ignored when evaluator uses run/example convention."""
        config = LangSmithEvaluatorConfig(
            evaluator="nat.plugins.langchain.eval.langsmith_evaluator._detect_convention",
            extra_fields={"context": "ctx_field"},
        )
        builder = make_mock_builder()

        with (patch(
                "nat.plugins.langchain.eval.langsmith_evaluator._import_evaluator",
                return_value=lambda run, example=None: {
                    "key": "k", "score": 1.0
                },
        ), ):
            async with register_langsmith_evaluator(config, builder) as info:
                assert info.evaluate_fn is not None

        assert "extra_fields will be ignored" in caplog.text
