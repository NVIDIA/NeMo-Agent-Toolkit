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
"""Utilities for installed-agent policy handling."""

from __future__ import annotations

from typing import Any


def resolve_local_install_policy(raw_policy: Any) -> str:
    """Normalize local install policy to one of: skip, allow."""
    if isinstance(raw_policy, bool):
        return "allow" if raw_policy else "skip"
    policy = str(raw_policy).strip().lower()
    if policy not in {"skip", "allow"}:
        raise ValueError(f"Invalid local_install_policy={raw_policy!r}. Expected one of: skip, allow.")
    return policy


def is_local_install_allowed(policy: str) -> bool:
    """Resolve whether host install is allowed under local install policy."""
    return policy == "allow"
