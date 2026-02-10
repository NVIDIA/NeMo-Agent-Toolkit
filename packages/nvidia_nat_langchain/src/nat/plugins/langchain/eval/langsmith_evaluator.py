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

import importlib
import inspect
import logging
from typing import Any

from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_evaluator
from nat.data_models.component_ref import LLMRef
from nat.data_models.evaluator import EvaluatorBaseConfig
from nat.data_models.retry_mixin import RetryMixin
from nat.utils.exception_handlers.automatic_retries import patch_with_retry

logger = logging.getLogger(__name__)


def _import_evaluator(dotted_path: str) -> Any:
    """Import an evaluator from a Python dotted path.

    Supports both module-level callables and class references:
    - ``'my_package.evaluators.my_function'`` -> imports and returns the function
    - ``'my_package.evaluators.MyClass'`` -> imports and instantiates the class

    Args:
        dotted_path: Full Python dotted path to the evaluator.

    Returns:
        The imported evaluator (callable or instance).

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the attribute cannot be found in the module.
    """
    parts = dotted_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid evaluator path '{dotted_path}'. Expected format: 'module.path.evaluator_name'")

    module_path, attr_name = parts

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}' for evaluator '{dotted_path}'. "
                          f"Make sure the package is installed and the path is correct.") from e

    evaluator = getattr(module, attr_name, None)
    if evaluator is None:
        raise AttributeError(f"Module '{module_path}' has no attribute '{attr_name}'. "
                             f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}")

    # If it's a class, instantiate it
    if isinstance(evaluator, type):
        try:
            evaluator = evaluator()
        except TypeError as e:
            raise TypeError(f"Could not instantiate class '{attr_name}' from '{dotted_path}'. "
                            f"If this class requires constructor arguments, instantiate it in "
                            f"your own code and use a factory function instead. Error: {e}") from e

    return evaluator


def _detect_convention(evaluator: Any) -> str:
    """Auto-detect which LangSmith evaluator convention is being used.

    Inspects the evaluator to determine if it's a RunEvaluator subclass,
    a function with ``(run, example)`` signature, or a function with
    ``(inputs, outputs, reference_outputs)`` signature.

    Args:
        evaluator: The evaluator callable or instance.

    Returns:
        One of ``'run_evaluator_class'``, ``'run_example_function'``,
        or ``'openevals_function'``.
    """
    # Check for RunEvaluator class instances (lazy import to avoid
    # pulling in langsmith at module load time)
    from langsmith.evaluation.evaluator import RunEvaluator

    if isinstance(evaluator, RunEvaluator):
        return "run_evaluator_class"

    # Inspect the callable's signature to determine convention
    if callable(evaluator):
        try:
            sig = inspect.signature(evaluator)
            param_names = [
                name for name, param in sig.parameters.items()
                if param.kind in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY, param.KEYWORD_ONLY)
            ]
        except (ValueError, TypeError):
            # If we can't inspect signature, default to openevals convention
            return "openevals_function"

        # Check for openevals-style: (inputs, outputs, reference_outputs)
        openevals_params = {"inputs", "outputs", "reference_outputs"}
        if openevals_params.intersection(param_names):
            return "openevals_function"

        # Check for LangSmith-style: (run, example)
        langsmith_params = {"run", "example"}
        if langsmith_params.intersection(param_names):
            return "run_example_function"

        # If the function takes positional args but we can't determine the convention,
        # try openevals first (more common in modern usage)
        return "openevals_function"

    raise ValueError(f"Cannot determine evaluator convention for {type(evaluator).__name__}. "
                     f"Expected a callable, RunEvaluator subclass, or function with "
                     f"(inputs, outputs, reference_outputs) or (run, example) signature.")


def _resolve_prompt(prompt_value: str) -> tuple[str, bool]:
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
        A tuple of ``(resolved_prompt_string, is_builtin)`` where
        ``is_builtin`` is ``True`` when the prompt was resolved from
        an openevals constant.
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
        return prompt_str, True

    # Not a known openevals prompt name -- treat as a literal prompt template
    return prompt_value, False


class LangSmithEvaluatorConfig(EvaluatorBaseConfig, RetryMixin, name="langsmith"):
    """Unified configuration for LangSmith evaluators.

    Supports two mutually exclusive modes:

    **Custom evaluator mode** (``evaluator`` field): imports any callable by
    dotted path. Supports RunEvaluator subclasses, ``(run, example)`` functions,
    and ``(inputs, outputs, reference_outputs)`` functions.

    **LLM-as-judge mode** (``prompt`` + ``llm_name`` fields): creates an
    openevals LLM-as-judge evaluator using a prebuilt or custom prompt and
    a judge LLM from the workflow's ``llms:`` section.
    """

    # -- Routing (exactly one of evaluator / prompt is required) --
    evaluator: str | None = Field(
        default=None,
        description="Python dotted path to a LangSmith evaluator callable "
        "(e.g., 'my_package.evaluators.my_fn' or 'openevals.exact_match'). "
        "Mutually exclusive with 'prompt'. When set, all other fields below are ignored.",
    )
    prompt: str | None = Field(
        default=None,
        description="Prebuilt openevals prompt name (e.g., 'correctness', 'hallucination') "
        "or a custom f-string prompt template for LLM-as-judge. "
        "Mutually exclusive with 'evaluator'. Requires 'llm_name'.",
    )

    # -- Prompt-mode fields (ignored when evaluator is set) --
    llm_name: LLMRef | None = Field(
        default=None,
        description="Name of the judge LLM from the workflow's llms: section. "
        "Required when 'prompt' is set; ignored when 'evaluator' is set.",
    )
    feedback_key: str = Field(
        default="score",
        description="Name under which the evaluation score is recorded. "
        "Appears as the metric column header in the LangSmith UI "
        "(e.g., 'correctness', 'helpfulness'). Prompt mode only.",
    )
    continuous: bool = Field(
        default=False,
        description="If True, score is a float between 0 and 1. "
        "If False, score is boolean. Mutually exclusive with 'choices'. Prompt mode only.",
    )
    choices: list[float] | None = Field(
        default=None,
        description="Explicit list of allowed score values (e.g., [0, 0.5, 1]). "
        "Mutually exclusive with 'continuous'. Prompt mode only.",
    )
    use_reasoning: bool = Field(
        default=True,
        description="If True, the judge model provides chain-of-thought reasoning "
        "alongside the score. Prompt mode only.",
    )

    @model_validator(mode="after")
    def _validate_mode(self) -> "LangSmithEvaluatorConfig":
        has_evaluator = self.evaluator is not None
        has_prompt = self.prompt is not None

        # -- Exactly one of evaluator / prompt must be set --
        if has_evaluator == has_prompt:
            raise ValueError("Exactly one of 'evaluator' or 'prompt' must be provided. "
                             "Use 'evaluator' to import a callable by dotted path, or "
                             "'prompt' to create an LLM-as-judge evaluator.")

        # -- Prompt mode requires llm_name --
        if has_prompt and self.llm_name is None:
            raise ValueError("'llm_name' is required when using 'prompt' mode. "
                             "Set it to the name of a judge LLM defined in the workflow's llms: section.")

        # -- continuous and choices are mutually exclusive --
        if self.continuous and self.choices is not None:
            raise ValueError("'continuous' and 'choices' are mutually exclusive. "
                             "Use 'continuous: true' for a float 0-1 score, or "
                             "'choices' for specific score values, but not both.")

        # -- Warn about prompt-mode-only fields in evaluator mode --
        if has_evaluator:
            prompt_only_overrides = []
            if self.llm_name is not None:
                prompt_only_overrides.append("llm_name")
            if self.feedback_key != "score":
                prompt_only_overrides.append("feedback_key")
            if self.continuous is not False:
                prompt_only_overrides.append("continuous")
            if self.choices is not None:
                prompt_only_overrides.append("choices")
            if self.use_reasoning is not True:
                prompt_only_overrides.append("use_reasoning")
            if prompt_only_overrides:
                logger.warning(
                    "Fields %s are only used in prompt mode and will be ignored "
                    "when 'evaluator' is set.",
                    prompt_only_overrides,
                )

        return self


@register_evaluator(config_type=LangSmithEvaluatorConfig)
async def register_langsmith_evaluator(config: LangSmithEvaluatorConfig, builder: EvalBuilder):
    """Register a LangSmith evaluator with NAT."""

    # Lazy import -- keeps langsmith out of the module-level import chain.
    from .langsmith_evaluator_adapter import LangSmithEvaluatorAdapter

    if config.evaluator is not None:
        # -- Custom evaluator mode: import callable, detect convention --
        evaluator_obj = _import_evaluator(config.evaluator)
        convention = _detect_convention(evaluator_obj)

        logger.info(
            "Loaded LangSmith evaluator '%s' (convention: %s)",
            config.evaluator,
            convention,
        )

        evaluator = LangSmithEvaluatorAdapter(
            evaluator=evaluator_obj,
            convention=convention,
            max_concurrency=builder.get_max_concurrency(),
            evaluator_name=config.evaluator.rsplit(".", 1)[-1],
        )

        yield EvaluatorInfo(
            config=config,
            evaluate_fn=evaluator.evaluate,
            description=f"LangSmith evaluator ({config.evaluator})",
        )

    else:
        # -- LLM-as-judge mode: resolve prompt, build judge, create evaluator --
        # The model_validator guarantees both are non-None here; assert for type narrowing.
        assert config.prompt is not None  # noqa: S101
        assert config.llm_name is not None  # noqa: S101

        try:
            from openevals.llm import create_llm_as_judge
        except ImportError as e:
            raise ImportError("The 'openevals' package is required for LLM-as-judge mode. "
                              "Install it with: pip install openevals") from e

        judge_llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

        if config.do_auto_retry:
            judge_llm = patch_with_retry(
                judge_llm,
                retries=config.num_retries,
                retry_codes=config.retry_on_status_codes,
                retry_on_messages=config.retry_on_errors,
            )

        resolved_prompt, is_builtin = _resolve_prompt(config.prompt)

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
            config.prompt.strip()[:50],
            config.llm_name,
        )

        # The LLM-as-judge callable follows the openevals convention
        evaluator = LangSmithEvaluatorAdapter(
            evaluator=evaluator_fn,
            convention="openevals_function",
            max_concurrency=builder.get_max_concurrency(),
            evaluator_name=config.feedback_key,
        )

        if is_builtin:
            desc = f"LangSmith '{config.prompt.strip().lower()}' LLM-as-judge (llm: {config.llm_name})"
        else:
            desc = f"LangSmith custom LLM-as-judge (llm: {config.llm_name})"

        yield EvaluatorInfo(config=config, evaluate_fn=evaluator.evaluate, description=desc)
