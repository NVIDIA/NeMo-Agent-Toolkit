# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

try:
    from adapter import DEFAULT_SOURCE_DATA
    from adapter import SimpleCalculatorPowerOfTwoAdapter
except ModuleNotFoundError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from adapter import DEFAULT_SOURCE_DATA
    from adapter import SimpleCalculatorPowerOfTwoAdapter

REPO_ROOT = Path(__file__).resolve().parents[5]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _default_output_dir() -> Path:
    return REPO_ROOT / "external" / "harbor" / "datasets" / "simple-calculator-power-of-two"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Harbor tasks for simple calculator power-of-two benchmark", )
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    adapter = SimpleCalculatorPowerOfTwoAdapter(task_dir=args.output_dir, source_file=args.source_file)
    task_ids = list(args.ids) if args.ids else adapter.list_available_tasks()
    if args.limit is not None:
        task_ids = task_ids[:max(0, args.limit)]
    generated, requested = adapter.generate_many(task_ids, overwrite=args.overwrite)
    logger.info("Generated %d/%d tasks in %s", generated, requested, args.output_dir)


if __name__ == "__main__":
    main()
