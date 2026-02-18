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

import logging
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from typing_extensions import is_typeddict

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.retry_mixin import RetryMixin

from .langsmith_evaluator import LangSmithExtraFieldsMixin

logger = logging.getLogger(__name__)


def _resolve_prompt(prompt_value: str) -> str:
    """Resolve a prompt name to the actual prompt string.

    Prompt names are resolved dynamically by convention: the short name is
    uppercased and suffixed with ``_PROMPT`` to form the constant name in
    ``openevals.prompts`` (e.g., ``'correctness'`` -> ``CORRECTNESS_PROMPT``).

    If the name doesn't match a constant in ``openevals.prompts``, it is
    treated as a literal prompt template string (e.g., a custom f-string).

    Args:
        prompt_value: A short prompt name (e.g., ``'correctness'``) or a literal prompt template string.

    Returns:
        The resolved prompt string.
    """
    normalized = prompt_value.strip().lower()
    constant_name = f"{normalized.upper()}_PROMPT"

    try:
        from openevals import prompts as openevals_prompts
    except ImportError as e:
        raise ImportError("The 'openevals' package is required to use LLM-as-judge prompts. "
                          "Install it with: pip install openevals") from e

    prompt_str = getattr(openevals_prompts, constant_name, None)
    if prompt_str is not None:
        return prompt_str

    # Not a known openevals prompt name -- treat as a literal prompt template
    return prompt_value


class LangSmithJudgeConfig(EvaluatorBaseConfig, RetryMixin, LangSmithExtraFieldsMixin, name="langsmith_judge"):
    """LLM-as-judge evaluator powered by openevals.

    Uses a prebuilt or custom prompt with a judge LLM to score workflow
    outputs. Prebuilt prompt names (e.g., ``'correctness'``, ``'hallucination'``)
    are resolved from openevals automatically.

    Common ``create_async_llm_as_judge`` parameters are exposed as typed fields
    for discoverability and validation.  Any additional / future parameters
    can be forwarded via the ``judge_kwargs`` pass-through dict.

    **Important:** The judge LLM must support structured output (JSON schema
    mode via ``with_structured_output``).  Models that do not support
    structured output will produce parsing errors and zero scores.  Verify
    that your chosen model supports this capability before use.
    """

    prompt: str = Field(description="Prebuilt openevals prompt name (e.g., 'correctness', 'hallucination') "
                        "or a custom f-string prompt template.", )
    llm_name: LLMRef = Field(description="Name of the judge LLM from the workflow's llms: section. "
                             "The model must support structured output (JSON schema mode).", )
    feedback_key: str = Field(
        default="score",
        description="Name under which the evaluation score is recorded. "
        "Appears as the metric column header in the LangSmith UI "
        "(e.g., 'correctness', 'helpfulness').",
    )
    continuous: bool = Field(
        default=False,
        description="If True, score is a float between 0 and 1. "
        "If False and 'choices' is not set, score is boolean. "
        "Mutually exclusive with 'choices'.",
    )
    choices: list[float] | None = Field(
        default=None,
        description="Explicit list of allowed score values (e.g., [0, 0.5, 1]). "
        "Mutually exclusive with 'continuous=True'.",
    )
    use_reasoning: bool = Field(
        default=True,
        description="If True, the judge model provides chain-of-thought reasoning "
        "alongside the score.",
    )
    system: str | None = Field(
        default=None,
        description="Optional system message prepended to the prompt. "
        "Only supported when 'prompt' is a string template.",
    )
    few_shot_examples: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional list of few-shot examples appended to the prompt "
        "to calibrate the judge. Each dict should have 'inputs', 'outputs', "
        "'score' (float or bool), and optionally 'reasoning' (str).",
    )
    output_schema: str | None = Field(
        default=None,
        description="Python dotted path to a TypedDict, Pydantic model, or other "
        "type accepted by openevals as a custom output schema "
        "(e.g., 'my_pkg.schemas.MyResult'). When set, the evaluator returns "
        "raw structured output matching the schema instead of the standard "
        "{key, score, comment} format.",
    )
    score_field: str = Field(
        default="score",
        description="Dot-notation path to the score field in custom output_schema "
        "results (e.g., 'analysis.score'). Only used when output_schema is set.",
    )
    judge_kwargs: dict[str, Any] | None = Field(
        default=None,
        description="Additional keyword arguments forwarded directly to "
        "openevals ``create_async_llm_as_judge``. Use this for parameters not "
        "exposed as typed fields. Keys must not overlap with typed fields.",
    )

    @model_validator(mode="after")
    def _validate_scoring(self) -> "LangSmithJudgeConfig":
        if self.continuous and self.choices is not None:
            raise ValueError("'continuous' and 'choices' are mutually exclusive. "
                             "Set continuous=True for a 0-1 float score, or provide "
                             "explicit 'choices', but not both.")
        return self


def _build_create_kwargs(
    config: LangSmithJudgeConfig,
    resolved_prompt: str,
    judge_llm: Any,
) -> dict[str, Any]:
    """Assemble keyword arguments for ``openevals.create_async_llm_as_judge``.

    Typed config fields are added first, then optional fields are merged
    only when set.  Finally, ``judge_kwargs`` is merged with overlap
    detection so that users cannot accidentally shadow typed fields.

    Args:
        config: The judge evaluator configuration.
        resolved_prompt: The prompt string, already resolved from a short name or left as-is for custom templates.
        judge_llm: The LLM instance to use as the judge.

    Returns:
        Dictionary of keyword arguments ready for ``create_async_llm_as_judge``.

    Raises:
        ValueError: If ``judge_kwargs`` keys overlap with typed fields.
    """
    from .utils import _import_from_dotted_path

    create_kwargs: dict[str, Any] = {
        "prompt": resolved_prompt,
        "judge": judge_llm,
        "feedback_key": config.feedback_key,
        "continuous": config.continuous,
        "choices": config.choices,
        "use_reasoning": config.use_reasoning,
    }

    if config.system is not None:
        create_kwargs["system"] = config.system

    if config.few_shot_examples is not None:
        create_kwargs["few_shot_examples"] = config.few_shot_examples

    if config.output_schema is not None:
        schema = _import_from_dotted_path(
            config.output_schema,
            label="output_schema",
        )

        if not (is_typeddict(schema) or (isinstance(schema, type) and issubclass(schema, BaseModel))):
            raise TypeError(f"output_schema must be a TypedDict or Pydantic BaseModel class, "
                            f"got {type(schema).__name__} from '{config.output_schema}'.")

        create_kwargs["output_schema"] = schema

    # Merge pass-through judge_kwargs, checking for overlap with the
    # typed fields that were already added to create_kwargs above.
    if config.judge_kwargs:
        overlap = set(create_kwargs) & set(config.judge_kwargs)
        if overlap:
            raise ValueError(f"judge_kwargs keys {overlap} overlap with typed config fields. "
                             f"Use the typed fields instead, or remove the overlapping keys "
                             f"from judge_kwargs.")
        create_kwargs.update(config.judge_kwargs)

    return create_kwargs


@register_evaluator(config_type=LangSmithJudgeConfig)
async def register_langsmith_judge(config: LangSmithJudgeConfig, builder: EvalBuilder):
    """Register an LLM-as-judge evaluator with NAT."""

    # Lazy imports -- keeps openevals and langsmith out of the module-level import chain.
    from openevals.llm import create_async_llm_as_judge

    from nat.utils.exception_handlers.automatic_retries import patch_with_retry

    from .langsmith_evaluator_adapter import LangSmithEvaluatorAdapter

    judge_llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    if config.do_auto_retry:
        judge_llm = patch_with_retry(
            judge_llm,
            retries=config.num_retries,
            retry_codes=config.retry_on_status_codes,
            retry_on_messages=config.retry_on_errors,
        )

    resolved_prompt = _resolve_prompt(config.prompt)
    create_kwargs = _build_create_kwargs(config, resolved_prompt, judge_llm)
    evaluator_fn = create_async_llm_as_judge(**create_kwargs)

    logger.info(
        "Created LLM-as-judge evaluator (prompt: %s, llm: %s)",
        config.prompt[:50],
        config.llm_name,
    )

    # Determine whether the adapter should use custom score_field parsing.
    # Only activate when a custom output_schema is set; otherwise the
    # standard result format is used and score_field is not needed.
    effective_score_field = config.score_field if config.output_schema is not None else None

    evaluator = LangSmithEvaluatorAdapter(
        evaluator=evaluator_fn,
        convention="openevals_function",
        max_concurrency=builder.get_max_concurrency(),
        evaluator_name=config.feedback_key,
        extra_fields=config.extra_fields,
        score_field=effective_score_field,
    )

    is_builtin = resolved_prompt != config.prompt
    if is_builtin:
        desc = f"LangSmith '{config.prompt.strip().lower()}' LLM-as-judge (llm: {config.llm_name})"
    else:
        desc = f"LangSmith custom LLM-as-judge (llm: {config.llm_name})"

    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description=desc)
