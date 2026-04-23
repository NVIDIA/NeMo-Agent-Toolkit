# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the Agentic Trajectory Observability Format (ATOF).

ATOF is a JSON-Lines wire format for agent runtime event streams. These
models define the two event kinds (``ScopeEvent``, ``MarkEvent``), the
behavioral flag enum (``Flags``), and the canonical ``category``
vocabulary (``Category``).

See ``atof-event-format.md`` for the core wire format.
"""

from nat.atof.category import Category
from nat.atof.events import Event
from nat.atof.events import MarkEvent
from nat.atof.events import ScopeEvent
from nat.atof.flags import Flags
from nat.atof.io import read_jsonl
from nat.atof.io import write_jsonl
from nat.atof.schemas import SCHEMA_REGISTRY
from nat.atof.schemas import lookup_schema
from nat.atof.schemas import register_schema

__all__ = [
    "SCHEMA_REGISTRY",
    "Category",
    "Event",
    "Flags",
    "MarkEvent",
    "ScopeEvent",
    "lookup_schema",
    "read_jsonl",
    "register_schema",
    "write_jsonl",
]
