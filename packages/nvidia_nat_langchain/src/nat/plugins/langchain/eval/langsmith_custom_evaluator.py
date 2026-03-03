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

import inspect
import logging
from typing import Any

from pydantic import Field

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig

from .langsmith_evaluator import LangSmithExtraFieldsMixin
from .utils import _import_from_dotted_path

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
    evaluator = _import_from_dotted_path(dotted_path, label="evaluator")

    # If it's a class, instantiate it
    if isinstance(evaluator, type):
        try:
            evaluator = evaluator()
        except TypeError as e:
            attr_name = dotted_path.rsplit(".", 1)[-1]
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
        One of ``'run_evaluator_class'``, ``'run_example_function'``, or ``'openevals_function'``.
    """
    # Check for RunEvaluator class instances (lazy import to avoid
    # pulling in langsmith at module load time)
    from langsmith.evaluation.evaluator import RunEvaluator

    from .langsmith_evaluator_adapter import _EvaluatorConvention

    if isinstance(evaluator, RunEvaluator):
        return _EvaluatorConvention.RUN_EVALUATOR_CLASS

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
            return _EvaluatorConvention.OPENEVALS_FUNCTION

        # Check for openevals-style: (inputs, outputs, reference_outputs)
        openevals_params = {"inputs", "outputs", "reference_outputs"}
        if openevals_params.intersection(param_names):
            return _EvaluatorConvention.OPENEVALS_FUNCTION

        # Check for LangSmith-style: (run, example)
        langsmith_params = {"run", "example"}
        if langsmith_params.intersection(param_names):
            return _EvaluatorConvention.RUN_EXAMPLE_FUNCTION

        # If the function has inspectable params but none match either convention,
        # default to openevals (more common in modern usage) and warn.
        logger.warning(
            "Could not determine evaluator convention from parameter names %s; "
            "defaulting to openevals (inputs, outputs, reference_outputs) convention.",
            param_names,
        )
        return _EvaluatorConvention.OPENEVALS_FUNCTION

    raise ValueError(f"Cannot determine evaluator convention for {type(evaluator).__name__}. "
                     f"Expected a callable, RunEvaluator subclass, or function with "
                     f"(inputs, outputs, reference_outputs) or (run, example) signature.")


class LangSmithCustomEvaluatorConfig(EvaluatorBaseConfig, LangSmithExtraFieldsMixin, name="langsmith_custom"):
    """Import any LangSmith-compatible evaluator by Python dotted path.

    Supports RunEvaluator subclasses, ``(run, example)`` functions,
    and ``(inputs, outputs, reference_outputs)`` functions. The calling
    convention is auto-detected at registration time.

    For built-in openevals evaluators, prefer ``_type: langsmith`` with a
    short name instead.
    """

    evaluator: str = Field(description="Python dotted path to a LangSmith evaluator callable "
                           "(e.g., 'my_package.evaluators.my_fn').", )


@register_evaluator(config_type=LangSmithCustomEvaluatorConfig)
async def register_langsmith_custom_evaluator(config: LangSmithCustomEvaluatorConfig, builder: EvalBuilder):
    """Register a custom LangSmith evaluator with NAT."""

    from .langsmith_evaluator_adapter import LangSmithEvaluatorAdapter

    evaluator_obj = _import_evaluator(config.evaluator)
    convention = _detect_convention(evaluator_obj)

    effective_extra_fields = config.extra_fields
    if config.extra_fields and convention != "openevals_function":
        logger.warning(
            "extra_fields is only supported with the openevals "
            "(inputs, outputs, reference_outputs) calling convention, but "
            "evaluator '%s' was detected as '%s'. "
            "extra_fields will be ignored for this evaluator.",
            config.evaluator,
            convention,
        )
        effective_extra_fields = None

    logger.info(
        "Loaded LangSmith custom evaluator '%s' (convention: %s)",
        config.evaluator,
        convention,
    )

    evaluator = LangSmithEvaluatorAdapter(
        evaluator=evaluator_obj,
        convention=convention,
        max_concurrency=builder.get_max_concurrency(),
        evaluator_name=config.evaluator.rsplit(".", 1)[-1],
        extra_fields=effective_extra_fields,
    )

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description=f"LangSmith custom evaluator ({config.evaluator})",
    )
