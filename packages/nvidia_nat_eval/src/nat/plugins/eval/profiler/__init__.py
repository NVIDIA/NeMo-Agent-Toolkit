# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Transitional profiler namespace aliases for eval package move."""

from __future__ import annotations

import importlib
import pkgutil
import sys

_LEGACY_PREFIX = "nat.profiler"
_NEW_PREFIX = "nat.plugins.eval.profiler"

_legacy_root = importlib.import_module(_LEGACY_PREFIX)


def _alias_module(old_name: str, new_name: str) -> None:
    if new_name in sys.modules:
        return
    sys.modules[new_name] = importlib.import_module(old_name)


def _populate_submodule_aliases() -> None:
    legacy_path = getattr(_legacy_root, "__path__", None)
    if legacy_path is None:
        return

    for module_info in pkgutil.walk_packages(legacy_path, prefix=f"{_LEGACY_PREFIX}."):
        # Keep `parameter_optimization` in core namespace; do not alias it under eval profiler.
        if ".parameter_optimization" in module_info.name:
            continue

        old_name = module_info.name
        new_name = old_name.replace(_LEGACY_PREFIX, _NEW_PREFIX, 1)
        _alias_module(old_name, new_name)


_populate_submodule_aliases()


def __getattr__(name: str):
    if name == "parameter_optimization":
        raise AttributeError(name)
    return getattr(_legacy_root, name)


def __dir__() -> list[str]:
    return sorted([name for name in dir(_legacy_root) if name != "parameter_optimization"])

