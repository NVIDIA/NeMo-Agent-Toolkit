# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections.abc import Sequence
from typing import Any
from typing import Generic
from typing import TypeVar

from optuna import Trial
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

T = TypeVar("T", int, float, bool, str)


# --------------------------------------------------------------------- #
# 1.  Hyper‑parameter metadata container                                #
# --------------------------------------------------------------------- #
class SearchSpace(BaseModel, Generic[T]):
    values: Sequence[T] | None = None
    low: T | None = None
    high: T | None = None
    log: bool = False  # log scale
    step: float | None = None
    is_prompt: bool = False
    prompt: str | None = None  # prompt to optimize
    prompt_purpose: str | None = None  # purpose of the prompt

    model_config = ConfigDict(protected_namespaces=(), extra="forbid")

    @model_validator(mode="after")
    def validate_search_space_parameters(self):
        """Validate that either values is provided, or both high and low."""
        if self.values is not None:
            # If values is provided, we don't need high/low
            if self.high is not None or self.low is not None:
                raise ValueError("SearchSpace 'values' is mutually exclusive with 'high' and 'low'")
            return self

        return self

    # Helper for Optuna Trials
    def suggest(self, trial: Trial, name: str):
        if self.is_prompt:
            raise ValueError("Prompt optimization not currently supported using Optuna. "
                             "Use the genetic algorithm implementation instead.")
        if self.values is not None:
            return trial.suggest_categorical(name, self.values)
        if isinstance(self.low, int):
            return trial.suggest_int(name, self.low, self.high, log=self.log, step=self.step)
        return trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)


def OptimizableField(
    default: Any,
    *,
    space: SearchSpace | None = None,
    merge_conflict: str = "overwrite",
    **fld_kw,
):
    # 1. Pull out any user‑supplied extras (must be a dict)
    user_extra = fld_kw.pop("json_schema_extra", None) or {}
    if not isinstance(user_extra, dict):
        raise TypeError("`json_schema_extra` must be a mapping.")

    # 2. If the space is a prompt, ensure a concrete base prompt exists
    if space is not None and getattr(space, "is_prompt", False):
        if getattr(space, "prompt", None) is None:
            if default is None:
                raise ValueError("Prompt-optimized fields require a base prompt: provide a "
                                 "non-None field default or set space.prompt.")
            # Default prompt not provided in space; fall back to the field's default
            space.prompt = default

    # 3. Prepare our own metadata
    ours = {"optimizable": True}
    if space is not None:
        ours["search_space"] = space

    # 4. Merge with user extras according to merge_conflict policy
    intersect = ours.keys() & user_extra.keys()
    if intersect:
        if merge_conflict == "error":
            raise ValueError("`json_schema_extra` already contains reserved key(s): "
                             f"{', '.join(intersect)}")
        if merge_conflict == "keep":
            # remove the ones the user already set so we don't overwrite them
            ours = {k: v for k, v in ours.items() if k not in intersect}

    merged_extra = {**user_extra, **ours}  # ours wins if 'overwrite'

    # 5. Return a normal Pydantic Field with merged extras
    return Field(default, json_schema_extra=merged_extra, **fld_kw)


class OptimizableMixin(BaseModel):
    optimizable_params: list[str] = Field(default_factory=list,
                                          description="List of parameters that can be optimized.",
                                          exclude=True)

    search_space: dict[str, SearchSpace] = Field(
        default_factory=dict,
        description="Optional search space overrides for optimizable parameters.",
        exclude=True,
    )
