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

import logging
from typing import get_args
from typing import get_origin

from pydantic import BaseModel

from nat.data_models.optimizable import SearchSpace

logger = logging.getLogger(__name__)


def walk_optimizables(obj: BaseModel, path: str = "") -> dict[str, SearchSpace]:
    """
    Recursively build ``{flattened.path: SearchSpace}`` for every optimizable
    field inside *obj*.

    * Honors ``optimizable_params`` on any model that mixes in
      ``OptimizableMixin`` – only listed fields are kept.
    * If a model contains optimizable fields **but** omits
      ``optimizable_params``, we emit a warning and skip them.
    """
    spaces: dict[str, SearchSpace] = {}

    allowed_params_raw = getattr(obj, "optimizable_params", None)
    allowed_params = set(allowed_params_raw) if allowed_params_raw is not None else None
    overrides = getattr(obj, "search_space", {}) or {}
    has_optimizable_flag = False

    for name, fld in obj.model_fields.items():
        full = f"{path}.{name}" if path else name
        extra = fld.json_schema_extra or {}

        is_field_optimizable = extra.get("optimizable", False) or name in overrides
        has_optimizable_flag = has_optimizable_flag or is_field_optimizable

        # honour allow-list
        if allowed_params is not None and name not in allowed_params:
            continue

        # 1. plain optimizable field or override from config
        if is_field_optimizable:
            space = overrides.get(name, extra.get("search_space"))
            if space is None:
                logger.error(
                    "Field %s is marked optimizable but no search space was provided.",
                    full,
                )
                raise ValueError(f"Field {full} is marked optimizable but no search space was provided")
            spaces[full] = space

        value = getattr(obj, name, None)

        # 2. nested BaseModel
        if isinstance(value, BaseModel):
            spaces.update(walk_optimizables(value, full))

        # 3. dict[str, BaseModel] container
        elif isinstance(value, dict):
            for key, subval in value.items():
                if isinstance(subval, BaseModel):
                    spaces.update(walk_optimizables(subval, f"{full}.{key}"))

        # 4. static-type fallback for class-level annotations
        elif isinstance(obj, type):
            ann = fld.annotation
            if get_origin(ann) in (dict, dict):
                _, val_t = get_args(ann) or (None, None)
                if isinstance(val_t, type) and issubclass(val_t, BaseModel):
                    if allowed_params is None or name in allowed_params:
                        spaces[f"{full}.*"] = SearchSpace(low=None, high=None)  # sentinel

    if allowed_params is None and has_optimizable_flag:
        logger.warning(
            "Model %s contains optimizable fields but no `optimizable_params` "
            "were defined; these fields will be ignored.",
            obj.__class__.__name__,
        )
    return spaces
