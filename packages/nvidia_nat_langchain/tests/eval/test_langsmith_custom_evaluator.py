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
from pydantic import ValidationError

from nat.builder.evaluator import EvaluatorInfo
from nat.data_models.evaluator import EvalInput
from nat.plugins.langchain.eval.langsmith_custom_evaluator import LangSmithCustomEvaluatorConfig
from nat.plugins.langchain.eval.langsmith_custom_evaluator import register_langsmith_custom_evaluator

from .conftest import make_mock_builder
from .conftest import register_evaluator_ctx


async def _register(config, builder=None):
    """Drive the async context manager returned by register_langsmith_custom_evaluator."""
    return await register_evaluator_ctx(register_langsmith_custom_evaluator, config, builder)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


class TestConfigValidation:
    """Tests for LangSmithCustomEvaluatorConfig validation."""

    def test_evaluator_accepts_dotted_path(self):
        """Config with a dotted path is valid."""
        config = LangSmithCustomEvaluatorConfig(evaluator="my_package.module.my_fn")
        assert config.evaluator == "my_package.module.my_fn"

    def test_evaluator_accepts_any_path(self):
        """Evaluator accepts any string; import errors happen at registration."""
        config = LangSmithCustomEvaluatorConfig(evaluator="nonexistent.path")
        assert config.evaluator == "nonexistent.path"

    def test_evaluator_required(self):
        """Omitting 'evaluator' raises a validation error."""
        with pytest.raises(ValidationError):
            LangSmithCustomEvaluatorConfig()


# --------------------------------------------------------------------------- #
# Registration with dotted paths
# --------------------------------------------------------------------------- #


class TestCustomEvaluatorRegistration:
    """Tests driven through register_langsmith_custom_evaluator with a mock builder.

    Covers all scenarios where the evaluator is referenced by a real importable
    dotted path (prebuilt openevals functions) and error cases for bad paths.
    """

    async def test_openevals_exact_match(self, eval_input_matching, eval_input_non_matching):
        """openevals.exact_match registered and evaluated via dotted path."""
        config = LangSmithCustomEvaluatorConfig(evaluator="openevals.exact_match")
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
        config = LangSmithCustomEvaluatorConfig(evaluator="openevals.exact_match_async")
        builder = make_mock_builder()

        info = await _register(config, builder)

        output_match = await info.evaluate_fn(eval_input_matching)
        assert output_match.eval_output_items[0].score is True

        output_mismatch = await info.evaluate_fn(eval_input_non_matching)
        assert output_mismatch.eval_output_items[0].score is False

    async def test_multi_item(self, eval_input_multi_item):
        """Evaluator processes multiple items correctly through registration."""
        config = LangSmithCustomEvaluatorConfig(evaluator="openevals.exact_match")
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
        config = LangSmithCustomEvaluatorConfig(evaluator="openevals.exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)
        output = await info.evaluate_fn(EvalInput(eval_input_items=[]))

        assert output.eval_output_items == []
        assert output.average_score is None

    async def test_evaluator_info_metadata(self):
        """EvaluatorInfo returned by registration has correct config and description."""
        config = LangSmithCustomEvaluatorConfig(evaluator="openevals.exact_match")
        builder = make_mock_builder()

        info = await _register(config, builder)

        assert info.config is config
        assert "exact_match" in info.description

    async def test_nonexistent_module_raises(self):
        """Registration raises ImportError for a nonexistent module."""
        config = LangSmithCustomEvaluatorConfig(evaluator="nonexistent_package.foo")
        builder = make_mock_builder()

        with pytest.raises(ImportError, match="Could not import module"):
            await _register(config, builder)

    async def test_nonexistent_attribute_raises(self):
        """Registration raises AttributeError for a nonexistent attribute."""
        config = LangSmithCustomEvaluatorConfig(evaluator="json.nonexistent_function")
        builder = make_mock_builder()

        with pytest.raises(AttributeError, match="has no attribute"):
            await _register(config, builder)

    async def test_bad_path_format_raises(self):
        """Registration raises ValueError for a path without a dot."""
        config = LangSmithCustomEvaluatorConfig(evaluator="no_dot_in_path")
        builder = make_mock_builder()

        with pytest.raises(ValueError, match="Invalid evaluator path"):
            await _register(config, builder)

    async def test_class_requiring_args_raises(self):
        """Registration raises TypeError for classes needing constructor arguments."""
        config = LangSmithCustomEvaluatorConfig(evaluator="datetime.datetime")
        builder = make_mock_builder()

        with pytest.raises(TypeError, match="Could not instantiate class"):
            await _register(config, builder)


# --------------------------------------------------------------------------- #
# extra_fields
# --------------------------------------------------------------------------- #


class TestLangSmithCustomEvaluatorConfigExtraFields:
    """Tests for extra_fields on the custom evaluator config."""

    def test_extra_fields_default_none(self):
        config = LangSmithCustomEvaluatorConfig(evaluator="openevals.exact_match")
        assert config.extra_fields is None

    def test_extra_fields_accepted(self):
        config = LangSmithCustomEvaluatorConfig(
            evaluator="openevals.exact_match",
            extra_fields={"context": "ctx_field"},
        )
        assert config.extra_fields == {"context": "ctx_field"}

    async def test_extra_fields_with_non_openevals_convention_warns_and_drops(self, caplog):
        """extra_fields warns and is ignored when evaluator uses run/example convention."""
        config = LangSmithCustomEvaluatorConfig(
            evaluator="nat.plugins.langchain.eval.langsmith_custom_evaluator._detect_convention",
            extra_fields={"context": "ctx_field"},
        )
        builder = make_mock_builder()

        with patch(
                "nat.plugins.langchain.eval.langsmith_custom_evaluator._import_evaluator",
                return_value=lambda run, example=None: {
                    "key": "k", "score": 1.0
                },
        ):
            async with register_langsmith_custom_evaluator(config, builder) as info:
                assert info.evaluate_fn is not None

        assert "extra_fields will be ignored" in caplog.text
