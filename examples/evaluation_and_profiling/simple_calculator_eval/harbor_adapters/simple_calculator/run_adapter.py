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

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ADAPTERS_DIR = SCRIPT_DIR.parent
for import_path in (ADAPTERS_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from adapter import SimpleCalculatorAdapter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_output_dir() -> Path:
    return REPO_ROOT / "external" / "harbor" / "datasets" / "simple-calculator"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Harbor tasks for Simple Calculator benchmark", )
    parser.add_argument("--source-file", type=Path, default=SimpleCalculatorAdapter.DEFAULT_SOURCE_DATA)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Generate Harbor tasks from the simple calculator benchmark.

    Args:
        None.

    Returns:
        None.
    """
    parser = _build_parser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = SimpleCalculatorAdapter(task_dir=args.output_dir, source_file=args.source_file)
    available_task_ids = adapter.list_available_tasks()
    if args.ids:
        requested_task_ids = list(args.ids)
        unknown_task_ids = sorted(set(requested_task_ids) - set(available_task_ids), key=requested_task_ids.index)
        if unknown_task_ids:
            parser.error(f"Unknown task ID(s): {', '.join(unknown_task_ids)}")
        task_ids = requested_task_ids
    else:
        task_ids = available_task_ids
    if args.limit is not None:
        task_ids = task_ids[:max(0, args.limit)]
    generated, requested = adapter.generate_many(task_ids, overwrite=args.overwrite)
    logger.info("Generated %d/%d tasks in %s", generated, requested, args.output_dir)


if __name__ == "__main__":
    main()
