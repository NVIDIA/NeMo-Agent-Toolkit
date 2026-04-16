# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attribute flag types for ATOF events.

Each type is a :class:`enum.StrEnum` whose members carry lowercase string
values. Attributes serialize on the JSON wire as a canonical (sorted,
deduplicated) array of strings — for example, ``["parallel", "relocatable"]``.

The flag sets defined here are the canonical names recognized by the ATOF
spec; the set is implementation-defined and extensible. Consumers MUST
preserve unknown flag strings rather than discarding them.

See ATOF spec Section 4.
"""

from __future__ import annotations

from enum import StrEnum


class ScopeAttributes(StrEnum):
    """Canonical behavioral flags for ScopeStart/ScopeEnd events.

    See ATOF spec Section 4.1.
    """

    PARALLEL = "parallel"
    RELOCATABLE = "relocatable"


class LLMAttributes(StrEnum):
    """Canonical behavioral flags for LLMStart/LLMEnd events.

    See ATOF spec Section 4.2.
    """

    STATELESS = "stateless"
    STREAMING = "streaming"


class ToolAttributes(StrEnum):
    """Canonical behavioral flags for ToolStart/ToolEnd events.

    See ATOF spec Section 4.3.
    """

    LOCAL = "local"
