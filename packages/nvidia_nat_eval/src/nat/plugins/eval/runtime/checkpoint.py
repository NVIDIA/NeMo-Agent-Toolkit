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

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import IO

from pydantic import ValidationError

from nat.data_models.evaluator import EvalInputItem
from nat.data_models.intermediate_step import IntermediateStep

logger = logging.getLogger(__name__)

CHECKPOINT_FILE_NAME = "workflow_output_checkpoint.jsonl"

_REQUIRED_KEYS = {"id", "input", "output", "trajectory"}


def _key(item_id, item_input) -> tuple[str, str]:
    return str(item_id), str(item_input)


class EvalCheckpoint:
    """
    One JSON line per dataset item whose workflow run finished, written as soon as it finishes.

    `nat eval` only writes `workflow_output.json` once every item is done, so a run that is interrupted
    (Ctrl-C, a killed job, a crash) loses every answer it already paid for. The checkpoint keeps them, and
    `nat eval --resume` puts them back so only the unfinished items run again.

    A record is matched to a dataset item by its id and its input together, so an item whose question was
    edited since the checkpoint was written runs again, and datasets without an id column still match.
    """

    def __init__(self, path: Path):
        self.path = path
        self._file: IO[str] | None = None

    def start(self, resume: bool) -> None:
        """Open the checkpoint for writing; a fresh run starts it empty, a resumed run appends to it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a" if resume else "w", encoding="utf-8")  # noqa: SIM115 closed in close()
        if resume and self._ends_in_a_torn_line():
            # end the partial line so the next record starts on its own line
            self._file.write("\n")
            self._file.flush()

    def record(self, item: EvalInputItem) -> None:
        """Write one finished item and flush it, so it survives the process being stopped right after."""
        if self._file is None:
            return
        record = {
            "id": item.id,
            "input": item.input_obj,
            "output": item.output_obj,
            "trajectory": [step.model_dump(mode="json") for step in item.trajectory],
        }
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _ends_in_a_torn_line(self) -> bool:
        with open(self.path, "rb") as f:
            f.seek(0, 2)
            if f.tell() == 0:
                return False
            f.seek(-1, 2)
            return f.read(1) != b"\n"

    def _load(self) -> dict[tuple[str, str], list[dict]]:
        records: dict[tuple[str, str], list[dict]] = defaultdict(list)
        if not self.path.exists():
            logger.warning("No checkpoint found at %s; running every item", self.path)
            return records
        skipped = 0
        with open(self.path, "rb") as f:
            for raw_line in f:
                if not raw_line.strip():
                    continue
                try:
                    # decode per line: a run stopped mid-write can cut a multi-byte character
                    record = json.loads(raw_line.decode("utf-8"))
                    if not _REQUIRED_KEYS.issubset(record):
                        raise KeyError("missing fields")
                    records[_key(record["id"], record["input"])].append(record)
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
                    # a run stopped mid-write leaves a partial line
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d unreadable line(s) in checkpoint %s", skipped, self.path)
        return records

    def restore(self, items: list[EvalInputItem]) -> list[EvalInputItem]:
        """Fill in the output and trajectory of every item the checkpoint has; return those items."""
        records = self._load()
        restored = []
        for item in items:
            matches = records.get(_key(item.id, item.input_obj), [])
            while matches:
                record = matches.pop(0)
                try:
                    trajectory = [IntermediateStep.model_validate(step) for step in record["trajectory"]]
                except (ValidationError, TypeError):
                    logger.warning("Skipping a checkpoint record for item %r with an unreadable trajectory", item.id)
                    continue
                item.output_obj = record["output"]
                item.trajectory = trajectory
                restored.append(item)
                break
        return restored
