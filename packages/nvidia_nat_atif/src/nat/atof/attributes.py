# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attribute bitflag types for ATOF events.

Mirrors NeMo-Flow's Rust ``bitflags!`` definitions without importing them.
Each type is a Python :class:`enum.IntFlag` that serializes as a plain integer
in JSON (e.g., ``PARALLEL | RELOCATABLE`` → ``3``).
"""

from __future__ import annotations

from enum import IntFlag


class ScopeAttributes(IntFlag):
    """Behavioral flags for ScopeStart/ScopeEnd events.

    See ATOF spec Section 4.1.
    """

    NONE = 0
    PARALLEL = 1
    RELOCATABLE = 2


class LLMAttributes(IntFlag):
    """Behavioral flags for LLMStart/LLMEnd events.

    See ATOF spec Section 4.2.
    """

    NONE = 0
    STATELESS = 1
    STREAMING = 2


class ToolAttributes(IntFlag):
    """Behavioral flags for ToolStart/ToolEnd events.

    See ATOF spec Section 4.3.
    """

    NONE = 0
    LOCAL = 1
