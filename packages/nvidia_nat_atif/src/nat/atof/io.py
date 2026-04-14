# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATOF JSON-Lines I/O utilities.

Read and write ATOF event streams as JSON-Lines files (one JSON object per line).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nat.atof.events import Event

if TYPE_CHECKING:
    pass

_event_adapter = TypeAdapter(Event)


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
    trailing newline.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for event in events:
        # `serialize_as_any=True` is REQUIRED for vendor-field round-trip: `profile` is typed as
        # `ProfileContract | None` (base class, per D-22), and Pydantic v2's default "by attribute
        # typing" serialization would drop subclass fields (e.g., DefaultLlmV1.model_name,
        # DefaultToolV1.tool_call_id). With duck-typing enabled, the runtime-instance fields are
        # serialized. `by_alias=True` ensures `$schema`/`$version`/`$mode` keys are emitted with
        # their wire-format `$`-prefixed names rather than the Python attribute names.
        lines.append(json.dumps(event.model_dump(by_alias=True, exclude_none=True, mode="json", serialize_as_any=True)))
    path.write_text("\n".join(lines) + "\n")
