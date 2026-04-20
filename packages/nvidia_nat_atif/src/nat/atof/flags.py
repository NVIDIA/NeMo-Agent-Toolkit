# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common behavioral flags for ATOF scope events.

Serializes as a canonical (sorted, deduplicated) lowercase string array. The
vocabulary is shared across all scope types per spec §4.1; there are no
per-scope-type flag subclasses. Consumers MUST preserve unknown flag strings
rather than discarding them.

See ATOF spec Section 4.1.
"""

from __future__ import annotations

from enum import StrEnum


class Flags(StrEnum):
    """Canonical behavioral flags for ScopeStart/ScopeEnd events (spec §4.1)."""

    PARALLEL = "parallel"
    RELOCATABLE = "relocatable"
    STATELESS = "stateless"
    LOCAL = "local"
    STREAMING = "streaming"
