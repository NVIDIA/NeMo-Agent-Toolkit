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

from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import Sequence
from typing import TypeVar

from pydantic import BaseModel
from pydantic import Field

T = TypeVar("T", int, float, bool, str)


# --------------------------------------------------------------------- #
# 1.  Hyper‑parameter metadata container                                #
# --------------------------------------------------------------------- #
@dataclass(slots=True)
class SearchSpace(Generic[T]):
    from optuna import Trial

    low: T | Sequence[T] | None = None
    high: T | None = None
    log: bool = False  # log scale
    step: float = None
    is_prompt: bool = False
    prompt: str = None  # prompt to optimize
    prompt_purpose: str = None  # purpose of the prompt

    # Helper for Optuna Trials
    def suggest(self, trial: Trial, name: str):
        if self.is_prompt:
            raise ValueError("Prompt optimization not currently supported.")
        if self.high is None:
            return trial.suggest_categorical(name, self.low)
        if isinstance(self.low, int):
            return trial.suggest_int(name, self.low, self.high, log=self.log, step=self.step)
        return trial.suggest_float(name, self.low, self.high, log=self.log, step=self.step)


def OptimizableField(
    default: Any,
    *,
    space: SearchSpace,
    merge_conflict: str = "overwrite",
    **fld_kw,
):
    """
    Drop‑in replacement for `pydantic.Field` that stores optimisation
    metadata while respecting any user‑supplied `json_schema_extra`.

    Parameters
    ----------
    default : Any
        Usual first positional argument for Field(...).
    space : SearchSpace
        The optimiser range / distribution.
    merge_conflict : {'overwrite', 'error', 'keep'}
        Behaviour when the user's `json_schema_extra` already contains
        'optimizable' or 'search_space':
            • 'overwrite'  – replace with our values (default).
            • 'keep'       – leave user values untouched.
            • 'error'      – raise ValueError.
    **fld_kw
        The usual keyword arguments for `pydantic.Field`.

    Returns
    -------
    FieldInfo
        A Pydantic `Field` instance with merged metadata.
    """
    # 1. Pull out any user‑supplied extras (must be a dict)
    user_extra = fld_kw.pop("json_schema_extra", None) or {}
    if not isinstance(user_extra, dict):
        raise TypeError("`json_schema_extra` must be a mapping.")

    # 2. Prepare our own metadata
    ours = {"optimizable": True, "search_space": space}

    # 3. Merge with user extras according to merge_conflict policy
    intersect = ours.keys() & user_extra.keys()
    if intersect:
        if merge_conflict == "error":
            raise ValueError(f"`json_schema_extra` already contains reserved key(s): "
                             f"{', '.join(intersect)}")
        if merge_conflict == "keep":
            # remove the ones the user already set so we don't overwrite them
            ours = {k: v for k, v in ours.items() if k not in intersect}

    merged_extra = {**user_extra, **ours}  # ours wins if 'overwrite'

    # 4. Return a normal Pydantic Field with merged extras
    return Field(default, json_schema_extra=merged_extra, **fld_kw)


class OptimizableMixin(BaseModel):
    """
    Mixin for models that can be optimized.
    """
    optimizable_params: list[str] = Field(default_factory=list,
                                            description="List of parameters that can be optimized.")
