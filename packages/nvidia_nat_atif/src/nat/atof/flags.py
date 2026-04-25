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


"""Canonical attribute flags for ATOF scope events (spec §2.1).

Serializes as a canonical (sorted, deduplicated) lowercase string array. The
vocabulary is shared across all categories; applicability per category is
documented in spec §2.1. Consumers MUST preserve unknown flag strings when
re-emitting and MUST NOT treat unknown flags as errors — vendor extensions
following the ``vendor.name`` dotted-namespace convention are forward-compat.
"""

from __future__ import annotations

from enum import StrEnum


class Flags(StrEnum):
    """Canonical behavioral flags for scope events (spec §2.1).

    Each flag describes the exceptional runtime property of a scope; absence
    means the documented default applies.
    """

    PARALLEL = "parallel"  # applies to any category (default: serial)
    RELOCATABLE = "relocatable"  # applies to any category (default: pinned)
    STATEFUL = "stateful"  # applies primarily to category=='llm' (default: stateless)
    STREAMING = "streaming"  # applies primarily to category=='llm' (default: single-payload)
    REMOTE = "remote"  # applies primarily to category=='tool' (default: local)
