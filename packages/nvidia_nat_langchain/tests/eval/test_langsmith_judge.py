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
from pydantic import ValidationError

from nat.builder.evaluator import EvaluatorInfo
from nat.plugins.langchain.eval.langsmith_judge import LangSmithJudgeConfig
from nat.plugins.langchain.eval.langsmith_judge import register_langsmith_judge

from .conftest import make_mock_builder
from .conftest import register_evaluator_ctx


async def _register(config, builder):
    """Drive the async context manager returned by register_langsmith_judge."""
    return await register_evaluator_ctx(register_langsmith_judge, config, builder)


class TestLangSmithJudgeConfig:
    """Tests for LangSmithJudgeConfig validation."""

    def test_valid_config(self):
        """Config with prompt + llm_name is valid."""
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.prompt == "correctness"
        assert config.llm_name == "eval_llm"

    def test_defaults(self):
        """Default values are correct."""
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.feedback_key == "score"
        assert config.continuous is False
        assert config.choices is None
        assert config.use_reasoning is True

    def test_custom_options(self):
        """Custom scoring options are accepted."""
        config = LangSmithJudgeConfig(
            prompt="Is the output polite? {inputs} {outputs}",
            llm_name="eval_llm",
            feedback_key="politeness",
            continuous=True,
        )
        assert config.feedback_key == "politeness"
        assert config.continuous is True

    def test_missing_prompt_raises(self):
        """Omitting 'prompt' raises a validation error."""
        with pytest.raises(ValidationError):
            LangSmithJudgeConfig(llm_name="eval_llm")

    def test_missing_llm_name_raises(self):
        """Omitting 'llm_name' raises a validation error."""
        with pytest.raises(ValidationError):
            LangSmithJudgeConfig(prompt="correctness")

    def test_continuous_and_choices_raises(self):
        """continuous and choices are mutually exclusive."""
        with pytest.raises(ValueError, match="'continuous' and 'choices' are mutually exclusive"):
            LangSmithJudgeConfig(
                prompt="correctness",
                llm_name="eval_llm",
                continuous=True,
                choices=[0.0, 0.5, 1.0],
            )

    def test_choices_without_continuous_is_valid(self):
        """choices alone (without continuous) is valid."""
        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            choices=[0.0, 0.5, 1.0],
        )
        assert config.choices == [0.0, 0.5, 1.0]
        assert config.continuous is False


class TestLangSmithJudgeRegistration:
    """Tests for the LLM-as-judge registration path using a mocked LLM.

    These tests exercise register_langsmith_judge end-to-end
    without requiring a real LLM by mocking builder.get_llm and
    create_async_llm_as_judge.
    """

    async def test_builtin_prompt(self, eval_input_matching):
        """Builtin prompt name creates a working evaluator."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 0.9, "comment": "Looks good"}

        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        builder = make_mock_builder(mock_llm)

        with patch(
                "openevals.llm.create_async_llm_as_judge",
                return_value=fake_judge,
        ) as mock_create:
            info = await _register(config, builder)

            # Verify create_async_llm_as_judge was called with the resolved prompt
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

    async def test_custom_prompt(self, eval_input_matching):
        """Custom prompt template is passed through to create_async_llm_as_judge."""
        custom_prompt = "Rate professionalism: {inputs} {outputs}"
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "professionalism", "score": 0.85, "comment": "Professional tone"}

        config = LangSmithJudgeConfig(
            prompt=custom_prompt,
            llm_name="eval_llm",
            feedback_key="professionalism",
            continuous=True,
        )
        builder = make_mock_builder(mock_llm)

        with patch(
                "openevals.llm.create_async_llm_as_judge",
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

    async def test_with_choices(self, eval_input_matching):
        """Choices are passed through to create_async_llm_as_judge."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 0.5}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            choices=[0.0, 0.5, 1.0],
        )
        builder = make_mock_builder(mock_llm)

        with patch(
                "openevals.llm.create_async_llm_as_judge",
                return_value=fake_judge,
        ) as mock_create:
            info = await _register(config, builder)

            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["choices"] == [0.0, 0.5, 1.0]
            assert call_kwargs["continuous"] is False

        output = await info.evaluate_fn(eval_input_matching)
        assert output.eval_output_items[0].score == 0.5

    async def test_retry_applied(self):
        """When do_auto_retry is True, patch_with_retry is called on the LLM."""
        mock_llm = MagicMock(name="mock_judge_llm")
        patched_llm = MagicMock(name="patched_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            do_auto_retry=True,
            num_retries=5,
        )
        builder = make_mock_builder(mock_llm)

        with (
                patch(
                    "nat.utils.exception_handlers.automatic_retries.patch_with_retry",
                    return_value=patched_llm,
                ) as mock_retry,
                patch(
                    "openevals.llm.create_async_llm_as_judge",
                    return_value=fake_judge,
                ) as mock_create,
        ):
            await _register(config, builder)

            # Verify patch_with_retry was called with the original LLM
            mock_retry.assert_called_once()
            assert mock_retry.call_args[0][0] is mock_llm
            assert mock_retry.call_args[1]["retries"] == 5

            # Verify create_async_llm_as_judge received the patched LLM
            assert mock_create.call_args[1]["judge"] is patched_llm

    async def test_retry_not_applied_when_disabled(self):
        """When do_auto_retry is explicitly False, patch_with_retry is NOT called."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm", do_auto_retry=False)
        builder = make_mock_builder(mock_llm)

        with (
                patch("nat.utils.exception_handlers.automatic_retries.patch_with_retry") as mock_retry,
                patch(
                    "openevals.llm.create_async_llm_as_judge",
                    return_value=fake_judge,
                ),
        ):
            await _register(config, builder)
            mock_retry.assert_not_called()


# --------------------------------------------------------------------------- #
# system and few_shot_examples typed fields
# --------------------------------------------------------------------------- #


class TestLangSmithJudgeNewTypedFields:
    """Tests for system and few_shot_examples config fields."""

    def test_system_default_none(self):
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.system is None

    def test_system_accepted(self):
        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            system="You are a strict evaluator.",
        )
        assert config.system == "You are a strict evaluator."

    def test_few_shot_examples_default_none(self):
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.few_shot_examples is None

    def test_few_shot_examples_accepted(self):
        examples = [
            {
                "inputs": "Q1", "outputs": "A1", "score": True, "reasoning": "Good"
            },
            {
                "inputs": "Q2", "outputs": "A2", "score": 0.5, "reasoning": "Partial"
            },
        ]
        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            few_shot_examples=examples,
        )
        assert len(config.few_shot_examples) == 2
        assert config.few_shot_examples[0]["score"] is True

    async def test_system_passed_to_create_async_llm_as_judge(self):
        """system field is forwarded to create_async_llm_as_judge."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 0.9, "comment": "OK"}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            system="Be strict.",
        )
        builder = make_mock_builder(mock_llm)

        with patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge) as mock_create:
            await _register(config, builder)
            assert mock_create.call_args[1]["system"] == "Be strict."

    async def test_few_shot_passed_to_create_async_llm_as_judge(self):
        """few_shot_examples field is forwarded to create_async_llm_as_judge."""
        mock_llm = MagicMock(name="mock_judge_llm")
        examples = [{"inputs": "Q", "outputs": "A", "score": True, "reasoning": "OK"}]

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            few_shot_examples=examples,
        )
        builder = make_mock_builder(mock_llm)

        with patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge) as mock_create:
            await _register(config, builder)
            assert mock_create.call_args[1]["few_shot_examples"] == examples

    async def test_system_not_passed_when_none(self):
        """system is NOT in create_async_llm_as_judge kwargs when None."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        builder = make_mock_builder(mock_llm)

        with patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge) as mock_create:
            await _register(config, builder)
            assert "system" not in mock_create.call_args[1]
            assert "few_shot_examples" not in mock_create.call_args[1]


# --------------------------------------------------------------------------- #
# judge_kwargs pass-through
# --------------------------------------------------------------------------- #


class TestJudgeKwargs:
    """Tests for judge_kwargs pass-through and overlap validation."""

    def test_judge_kwargs_default_none(self):
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.judge_kwargs is None

    def test_judge_kwargs_accepted(self):
        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            judge_kwargs={"some_future_param": 42},
        )
        assert config.judge_kwargs == {"some_future_param": 42}

    async def test_judge_kwargs_overlap_with_typed_field_raises(self):
        """Overlap between judge_kwargs and typed fields raises ValueError at registration."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            judge_kwargs={"continuous": True},
        )
        builder = make_mock_builder(mock_llm)

        with (
                pytest.raises(ValueError, match="overlap with typed config fields"),
                patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge),
        ):
            await _register(config, builder)

    async def test_judge_kwargs_overlap_system_raises(self):
        """Overlap with system field raises when both typed field and judge_kwargs set it."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            system="typed value",
            judge_kwargs={"system": "duplicate value"},
        )
        builder = make_mock_builder(mock_llm)

        with (
                pytest.raises(ValueError, match="overlap with typed config fields"),
                patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge),
        ):
            await _register(config, builder)

    async def test_judge_kwargs_system_allowed_when_typed_field_unset(self):
        """system in judge_kwargs is fine when the typed field is None (no overlap)."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            judge_kwargs={"system": "via kwargs"},
        )
        builder = make_mock_builder(mock_llm)

        with patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge) as mock_create:
            await _register(config, builder)
            assert mock_create.call_args[1]["system"] == "via kwargs"

    async def test_judge_kwargs_forwarded_to_create_async_llm_as_judge(self):
        """judge_kwargs are merged into create_async_llm_as_judge call."""
        mock_llm = MagicMock(name="mock_judge_llm")

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            judge_kwargs={"some_future_param": "hello"},
        )
        builder = make_mock_builder(mock_llm)

        with patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge) as mock_create:
            await _register(config, builder)
            assert mock_create.call_args[1]["some_future_param"] == "hello"


# --------------------------------------------------------------------------- #
# output_schema and score_field
# --------------------------------------------------------------------------- #


class TestOutputSchema:
    """Tests for output_schema and score_field config fields."""

    def test_output_schema_default_none(self):
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.output_schema is None

    def test_output_schema_accepted(self):
        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            output_schema="my_pkg.schemas.MyResult",
        )
        assert config.output_schema == "my_pkg.schemas.MyResult"

    def test_score_field_default(self):
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.score_field == "score"

    def test_score_field_custom(self):
        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            output_schema="my_pkg.schemas.MyResult",
            score_field="analysis.score",
        )
        assert config.score_field == "analysis.score"

    async def test_output_schema_imported_and_passed(self):
        """output_schema dotted path is imported and passed to create_async_llm_as_judge."""
        from typing import TypedDict

        mock_llm = MagicMock(name="mock_judge_llm")

        class FakeResult(TypedDict):
            score: float
            reasoning: str

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            output_schema="my_pkg.schemas.FakeResult",
        )
        builder = make_mock_builder(mock_llm)

        with (
                patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge) as mock_create,
                patch(
                    "nat.plugins.langchain.eval.utils._import_from_dotted_path",
                    return_value=FakeResult,
                ) as mock_import,
        ):
            await _register(config, builder)
            mock_import.assert_called_once_with("my_pkg.schemas.FakeResult", label="output_schema")
            assert mock_create.call_args[1]["output_schema"] is FakeResult

    async def test_output_schema_rejects_non_typeddict_non_basemodel(self):
        """output_schema that is not a TypedDict or BaseModel raises TypeError."""
        mock_llm = MagicMock(name="mock_judge_llm")
        invalid_schema = type("NotASchema", (), {})

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            return {"key": "score", "score": 1.0}

        config = LangSmithJudgeConfig(
            prompt="correctness",
            llm_name="eval_llm",
            output_schema="my_pkg.schemas.NotASchema",
        )
        builder = make_mock_builder(mock_llm)

        with (
                pytest.raises(TypeError, match="must be a TypedDict or Pydantic BaseModel class"),
                patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge),
                patch(
                    "nat.plugins.langchain.eval.utils._import_from_dotted_path",
                    return_value=invalid_schema,
                ),
        ):
            await _register(config, builder)


# --------------------------------------------------------------------------- #
# extra_fields on langsmith_judge
# --------------------------------------------------------------------------- #


class TestLangSmithJudgeExtraFields:
    """Tests for extra_fields on langsmith_judge."""

    def test_extra_fields_default_none(self):
        config = LangSmithJudgeConfig(prompt="correctness", llm_name="eval_llm")
        assert config.extra_fields is None

    def test_extra_fields_accepted(self):
        config = LangSmithJudgeConfig(
            prompt="hallucination",
            llm_name="eval_llm",
            extra_fields={"context": "retrieved_context"},
        )
        assert config.extra_fields == {"context": "retrieved_context"}

    async def test_extra_fields_forwarded_through_adapter(self, eval_input_with_context):
        """extra_fields on langsmith_judge flow through to evaluator call."""
        mock_llm = MagicMock(name="mock_judge_llm")
        received = {}

        async def fake_judge(*, inputs=None, outputs=None, reference_outputs=None, **kwargs):  # noqa: ARG001
            received.update(kwargs)
            return {"key": "score", "score": True, "comment": "No hallucination"}

        config = LangSmithJudgeConfig(
            prompt="hallucination",
            llm_name="eval_llm",
            extra_fields={"context": "retrieved_context"},
        )
        builder = make_mock_builder(mock_llm)

        with patch("openevals.llm.create_async_llm_as_judge", return_value=fake_judge):
            info = await _register(config, builder)

        output = await info.evaluate_fn(eval_input_with_context)
        assert output.eval_output_items[0].score is True
        assert received["context"] == "Doodads are small mechanical gadgets used in workshops."
