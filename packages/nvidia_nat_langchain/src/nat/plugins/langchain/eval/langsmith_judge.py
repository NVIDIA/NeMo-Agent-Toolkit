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

from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.retry_mixin import RetryMixin

logger = logging.getLogger(__name__)


def _resolve_prompt(prompt_value: str) -> str:
    """Resolve a prompt name to the actual prompt string.

    Prompt names are resolved dynamically by convention: the short name is
    uppercased and suffixed with ``_PROMPT`` to form the constant name in
    ``openevals.prompts`` (e.g., ``'correctness'`` -> ``CORRECTNESS_PROMPT``).
    This means any prompt that openevals ships is automatically available
    without maintaining a hardcoded mapping.

    If the name doesn't match a constant in ``openevals.prompts``, it is
    treated as a literal prompt template string (e.g., a custom f-string).

    Args:
        prompt_value: Either a short prompt name (e.g., ``'correctness'``,
            ``'hallucination'``) or a literal prompt template string.

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


class LangSmithJudgeConfig(EvaluatorBaseConfig, RetryMixin, name="langsmith_judge"):
    """LLM-as-judge evaluator powered by openevals.

    Uses a prebuilt or custom prompt with a judge LLM to score workflow
    outputs. Prebuilt prompt names (e.g., ``'correctness'``, ``'hallucination'``)
    are resolved from openevals automatically.
    """

    prompt: str = Field(description="Prebuilt openevals prompt name (e.g., 'correctness', 'hallucination') "
                        "or a custom f-string prompt template.", )
    llm_name: LLMRef = Field(description="Name of the judge LLM from the workflow's llms: section.", )
    feedback_key: str = Field(
        default="score",
        description="Name under which the evaluation score is recorded. "
        "Appears as the metric column header in the LangSmith UI "
        "(e.g., 'correctness', 'helpfulness').",
    )
    continuous: bool = Field(
        default=False,
        description="If True, score is a float between 0 and 1. "
        "If False, score is boolean. Mutually exclusive with 'choices'.",
    )
    choices: list[float] | None = Field(
        default=None,
        description="Explicit list of allowed score values (e.g., [0, 0.5, 1]). "
        "Mutually exclusive with 'continuous'.",
    )
    use_reasoning: bool = Field(
        default=True,
        description="If True, the judge model provides chain-of-thought reasoning "
        "alongside the score.",
    )

    @model_validator(mode="after")
    def _validate_scoring(self) -> "LangSmithJudgeConfig":
        if self.continuous and self.choices is not None:
            raise ValueError("'continuous' and 'choices' are mutually exclusive. "
                             "Set continuous=True for a 0-1 float score, or provide "
                             "explicit 'choices', but not both.")
        return self


@register_evaluator(config_type=LangSmithJudgeConfig)
async def register_langsmith_judge(config: LangSmithJudgeConfig, builder: EvalBuilder):
    """Register an LLM-as-judge evaluator with NAT."""

    # Lazy imports -- keeps openevals and langsmith out of the module-level import chain.
    from openevals.llm import create_llm_as_judge

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

    evaluator_fn = create_llm_as_judge(
        prompt=resolved_prompt,
        judge=judge_llm,
        feedback_key=config.feedback_key,
        continuous=config.continuous,
        choices=config.choices,
        use_reasoning=config.use_reasoning,
    )

    logger.info(
        "Created LLM-as-judge evaluator (prompt: %s, llm: %s)",
        config.prompt[:50],
        config.llm_name,
    )

    # The LLM-as-judge callable follows the openevals convention
    evaluator = LangSmithEvaluatorAdapter(
        evaluator=evaluator_fn,
        convention="openevals_function",
        max_concurrency=builder.get_max_concurrency(),
        evaluator_name=config.feedback_key,
    )

    is_builtin = resolved_prompt != config.prompt
    if is_builtin:
        desc = f"LangSmith '{config.prompt.strip().lower()}' LLM-as-judge (llm: {config.llm_name})"
    else:
        desc = f"LangSmith custom LLM-as-judge (llm: {config.llm_name})"

    yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description=desc)
