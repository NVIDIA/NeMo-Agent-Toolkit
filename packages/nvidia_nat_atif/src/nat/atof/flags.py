# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical attribute flags for ATOF scope events (spec §2.1).

Serializes as a canonical (sorted, deduplicated) lowercase string array. The
vocabulary is shared across all scope_types; applicability per scope_type is
documented in spec §2.1. Consumers MUST preserve unknown flag strings when
re-emitting and MUST NOT treat unknown flags as errors — vendor extensions
following the ``vendor.name`` dotted-namespace convention are forward-compat.
"""

from __future__ import annotations

from enum import StrEnum


class Flags(StrEnum):
    """Canonical behavioral flags for ScopeStart/ScopeEnd events (spec §2.1).

    Each flag describes the exceptional runtime property of a scope; absence
    means the documented default applies.
    """

    PARALLEL = "parallel"  # applies to any scope_type (default: serial)
    RELOCATABLE = "relocatable"  # applies to any scope_type (default: pinned)
    STATEFUL = "stateful"  # applies primarily to scope_type=='llm' (default: stateless)
    STREAMING = "streaming"  # applies primarily to scope_type=='llm' (default: single-payload)
    REMOTE = "remote"  # applies primarily to scope_type=='tool' (default: local)
