# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Deprecated compatibility shim for `nat.eval`.

Evaluation modules moved to `nat.plugins.eval`. This shim keeps old import
paths working for one release cycle.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

warnings.warn(
    "Importing from 'nat.eval' is deprecated and will be removed in a future release. "
    "Use 'nat.plugins.eval' instead.",
    UserWarning,
    stacklevel=2,
)

_NEW_PREFIX = "nat.plugins.eval"
_OLD_PREFIX = "nat.eval"

_new_root = importlib.import_module(_NEW_PREFIX)


def _alias_module(old_name: str, new_name: str) -> None:
    if old_name in sys.modules:
        return
    try:
        sys.modules[old_name] = importlib.import_module(new_name)
    except ImportError:
        # Some eval submodules depend on optional third-party packages. Skip
        # aliasing those modules so importing `nat.eval` still works.
        return


def _populate_aliases() -> None:
    _alias_module(_OLD_PREFIX, _NEW_PREFIX)
    new_path = getattr(_new_root, "__path__", None)
    if new_path is None:
        return

    for module_info in pkgutil.walk_packages(new_path, prefix=f"{_NEW_PREFIX}."):
        new_name = module_info.name
        old_name = new_name.replace(_NEW_PREFIX, _OLD_PREFIX, 1)
        _alias_module(old_name, new_name)


_populate_aliases()
_public_names = getattr(_new_root, "__all__", None)
if _public_names is None:
    _public_names = [name for name in dir(_new_root) if not name.startswith("_")]

globals().update({name: getattr(_new_root, name) for name in _public_names})
__all__ = list(_public_names)
