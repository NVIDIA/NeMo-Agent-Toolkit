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
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.builder.builder import EvalBuilder
from nat.builder.evaluator import EvaluatorInfo
from nat.cli.register_workflow import register_evaluator
from nat.data_models.evaluator import EvaluatorBaseConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------


def _get_registry() -> dict[str, Callable[..., Any]]:
    """Return the evaluator registry, importing openevals lazily.

    Keeps openevals out of the module-level import chain while providing
    a single source of truth for known evaluator names and their callables.

    Async variants are used to align with NAT's async-first design.
    The adapter (:class:`LangSmithEvaluatorAdapter`) awaits async
    callables directly via ``_invoke_maybe_sync``, avoiding unnecessary
    thread-pool dispatch.
    """
    from openevals import exact_match_async
    from openevals.string import levenshtein_distance_async

    return {
        "exact_match": exact_match_async,
        "levenshtein_distance": levenshtein_distance_async,
    }


def _resolve_evaluator(name: str) -> Callable[..., Any]:
    """Resolve a short evaluator name to its openevals callable.

    The model validator on :class:`LangSmithEvaluatorConfig` already
    ensures *name* is valid, so this is a direct lookup.

    Args:
        name: Short evaluator name (e.g., ``'exact_match'``, ``'levenshtein_distance'``).

    Returns:
        The resolved evaluator callable.
    """
    return _get_registry()[name]


class LangSmithExtraFieldsMixin(BaseModel):
    """Mixin for extra fields on the LangSmith evaluator config."""
    extra_fields: dict[str, str] | None = Field(
        default=None,
        description="Optional mapping of evaluator kwarg names to dataset field names.  "
        "Keys are the kwarg names passed to the evaluator; values are looked up "
        "in the dataset entry.  Example: ``{context: retrieved_context}`` passes "
        "the dataset's 'retrieved_context' field as the 'context' kwarg.",
    )


class LangSmithEvaluatorConfig(EvaluatorBaseConfig, LangSmithExtraFieldsMixin, name="langsmith"):
    """Built-in openevals evaluator selected by short name.

    Resolves evaluator names (e.g., ``'exact_match'``,
    ``'levenshtein_distance'``) from the openevals package automatically.
    For custom user-defined evaluators, use ``_type: langsmith_custom``
    instead.
    """

    evaluator: str = Field(description="Short name of an openevals evaluator "
                           "(e.g., 'exact_match', 'levenshtein_distance').", )

    @model_validator(mode="after")
    def _validate_evaluator_name(self) -> "LangSmithEvaluatorConfig":
        """Validate that the evaluator name exists in the registry."""
        registry = _get_registry()
        if self.evaluator not in registry:
            raise ValueError(f"Unknown evaluator '{self.evaluator}'. "
                             f"Available evaluators: {sorted(registry.keys())}. "
                             f"For custom evaluators, use '_type: langsmith_custom' with a "
                             f"Python dotted path instead.")
        return self


@register_evaluator(config_type=LangSmithEvaluatorConfig)
async def register_langsmith_evaluator(config: LangSmithEvaluatorConfig, builder: EvalBuilder):
    """Register a built-in openevals evaluator with NAT."""

    from .langsmith_evaluator_adapter import LangSmithEvaluatorAdapter

    evaluator_fn = _resolve_evaluator(config.evaluator)

    logger.info(
        "Loaded LangSmith evaluator '%s' (convention: openevals_function)",
        config.evaluator,
    )

    evaluator = LangSmithEvaluatorAdapter(
        evaluator=evaluator_fn,
        convention="openevals_function",
        max_concurrency=builder.get_max_concurrency(),
        evaluator_name=config.evaluator,
        extra_fields=config.extra_fields,
    )

    yield EvaluatorInfo(
        config=config,
        evaluate_fn=evaluator.evaluate,
        description=f"LangSmith evaluator ({config.evaluator})",
    )
