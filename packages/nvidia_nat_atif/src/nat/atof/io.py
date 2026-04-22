# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF JSON-Lines I/O utilities.

Read and write ATOF event streams as JSON-Lines files (one JSON object per line).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from nat.atof.events import Event

_event_adapter = TypeAdapter(Event)


# Canonical on-wire key order per ATOF v0.1 spec. Scope events follow the
# order in §3.1's field table; mark events follow §3.2's. Any key the
# converter receives that isn't in these lists is appended in
# insertion-order, which preserves vendor extensions under
# ``ConfigDict(extra="allow")``.
_SCOPE_WIRE_ORDER = (
    "kind",
    "scope_category",
    "atof_version",
    "category",
    "category_profile",
    "uuid",
    "parent_uuid",
    "data",
    "data_schema",
    "timestamp",
    "name",
    "attributes",
    "metadata",
)
_MARK_WIRE_ORDER = (
    "kind",
    "atof_version",
    "category",
    "category_profile",
    "uuid",
    "parent_uuid",
    "data",
    "data_schema",
    "timestamp",
    "name",
    "metadata",
)


def _reorder_for_wire(event_dict: dict) -> dict:
    """Reorder a serialized event dict to match the ATOF spec §3 field order.

    Pydantic's ``model_dump`` emits subclass fields after inherited base
    fields, which pushes ``kind``/``scope_category`` to the end of the
    output. The spec wire envelope example (§1) puts them first for
    readability. This reorders the dict while preserving any unknown keys
    (vendor extensions) at the end in insertion order.
    """
    kind = event_dict.get("kind")
    order = _SCOPE_WIRE_ORDER if kind == "scope" else _MARK_WIRE_ORDER
    ordered: dict = {}
    for key in order:
        if key in event_dict:
            ordered[key] = event_dict[key]
    for key, value in event_dict.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def read_jsonl(path: str | Path) -> list[Event]:
    """Read an ATOF JSON-Lines file and return a list of typed Event objects.

    Each line is parsed as a JSON object and validated against the Event
    discriminated union. Blank lines are skipped. Events are returned sorted
    by ``.ts_micros`` (the normalized int-microsecond timestamp, spec §5.1)
    so downstream consumers get a stable ordering across mixed str/int
    timestamp streams.
    """
    path = Path(path)
    events: list[Event] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        events.append(_event_adapter.validate_python(raw))
    events.sort(key=lambda e: e.ts_micros)
    return events


def write_jsonl(events: list[Event], path: str | Path) -> None:
    """Write a list of Event objects to a JSON-Lines file.

    Each event is serialized as a single JSON line. The file ends with a
    trailing newline. Optional fields with ``None`` values are emitted as
    explicit ``null`` on the wire (matching the spec wire envelope example
    in atof-event-format.md §1).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for event in events:
        # Exclude the computed ``ts_micros`` field from wire output — it's an
        # in-memory sorting convenience, not part of the wire envelope (spec §2).
        dumped = event.model_dump(exclude={"ts_micros"}, mode="json", by_alias=True)
        # Reorder to match the spec field tables (§3.1 scope, §3.2 mark) so
        # ``kind`` and ``scope_category`` lead the envelope.
        ordered = _reorder_for_wire(dumped)
        lines.append(json.dumps(ordered))
    path.write_text("\n".join(lines) + "\n")
