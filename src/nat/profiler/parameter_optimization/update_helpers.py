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

from collections import defaultdict
from typing import Any

from pydantic import BaseModel


def _deep_merge_dict(target: dict[str, Any], updates: dict[str, Any]) -> None:
    """In-place deep merge of nested dictionaries."""
    for key, value in updates.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            _deep_merge_dict(target[key], value)
        else:
            target[key] = value


def nest_updates(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Convert ``{'a.b.c': 1, 'd.x.y': 2}`` ➜
    ``{'a': {'b': {'c': 1}}, 'd': {'x': {'y': 2}}}``.
    Works even when the middle segment is a dict key.
    """
    root: dict[str, Any] = defaultdict(dict)

    for dotted, value in flat.items():
        head, *rest = dotted.split(".", 1)
        if not rest:  # leaf
            root[head] = value
            continue

        tail = rest[0]
        child_updates = nest_updates({tail: value})
        if isinstance(root[head], dict):
            _deep_merge_dict(root[head], child_updates)
        else:
            root[head] = child_updates
    return dict(root)


def apply_suggestions(cfg: BaseModel, flat: dict[str, Any]) -> BaseModel:
    """
    Return a **new** config where only the dotted-path keys in *flat*
    have been modified. Preserves all unrelated siblings.
    """
    cfg_dict = cfg.model_dump(mode="python")
    for dotted, value in flat.items():
        keys = dotted.split(".")
        cursor = cfg_dict
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return cfg.__class__.model_validate(cfg_dict)
