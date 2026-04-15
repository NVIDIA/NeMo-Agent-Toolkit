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
    discriminated union. Blank lines are skipped.
    """
    path = Path(path)
    events: list[Event] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        events.append(_event_adapter.validate_python(raw))
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
        lines.append(json.dumps(event.model_dump(exclude_none=True, mode="json")))
    path.write_text("\n".join(lines) + "\n")
