# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common behavioral flags for ATOF scope events.

Serializes as a canonical (sorted, deduplicated) lowercase string array. The
vocabulary is shared across all scope types per spec §2.1; there are no
per-scope-type flag subclasses. Each flag names the exceptional case — absence
means the documented default applies. Consumers MUST preserve unknown flag
strings rather than discarding them.

See ATOF spec Section 2.1.
"""

from __future__ import annotations

from enum import StrEnum


class Flags(StrEnum):
    """Canonical behavioral flags for ScopeStart/ScopeEnd events (spec §2.1).

    Each flag describes the exceptional runtime property of a scope; absence
    indicates the default. Defaults: parallel→serial, relocatable→pinned,
    stateful→stateless, remote→local, streaming→single-payload.
    """

    PARALLEL = "parallel"
    RELOCATABLE = "relocatable"
    STATEFUL = "stateful"
    REMOTE = "remote"
    STREAMING = "streaming"
