# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BFCL dataset loader.

Loads BFCL v3 JSONL test files into a DataFrame compatible with NAT's eval runner.
Each row contains the user question, function schemas, and ground truth answer.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader

from .config import BFCLDatasetConfig

logger = logging.getLogger(__name__)


def _resolve_possible_answer_path(test_file: Path, test_category: str) -> Path | None:
    """Auto-resolve the possible_answer file from the bfcl package."""
    # Try sibling directory
    possible_dir = test_file.parent / "possible_answer"
    candidate = possible_dir / test_file.name
    if candidate.is_file():
        return candidate

    # Try bfcl package data
    try:
        from bfcl.constant import POSSIBLE_ANSWER_PATH
        candidate = Path(POSSIBLE_ANSWER_PATH) / test_file.name
        if candidate.is_file():
            return candidate
    except ImportError:
        pass

    return None


def load_bfcl_dataset(file_path: str | Path, test_category: str = "simple") -> pd.DataFrame:
    """Load a BFCL v3 JSONL file into a DataFrame.

    Each row gets:
    - id: the BFCL test ID (e.g. "simple_0")
    - question: serialized JSON with the full test entry (question + function schemas)
    - answer: serialized JSON with the ground truth possible answer
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise ValueError(f"BFCL dataset file not found: {file_path}")

    # Load test prompts
    with open(file_path, encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    # Load possible answers
    answer_path = _resolve_possible_answer_path(file_path, test_category)
    answers_by_id = {}
    if answer_path and answer_path.is_file():
        with open(answer_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ans = json.loads(line)
                    answers_by_id[ans["id"]] = ans
        logger.info("Loaded %d possible answers from %s", len(answers_by_id), answer_path)
    else:
        logger.warning("No possible_answer file found for %s — evaluator will have no ground truth", file_path.name)

    rows = []
    for entry in entries:
        entry_id = entry["id"]
        answer = answers_by_id.get(entry_id, {"id": entry_id, "ground_truth": []})
        rows.append({
            "id": entry_id,
            "question": json.dumps(entry),
            "answer": json.dumps(answer),
        })

    if not rows:
        raise ValueError(f"No entries found in BFCL dataset: {file_path}")

    logger.info("Loaded %d BFCL entries from %s (category: %s)", len(rows), file_path.name, test_category)
    return pd.DataFrame(rows)


@register_dataset_loader(config_type=BFCLDatasetConfig)
async def register_bfcl_dataset_loader(config: BFCLDatasetConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(
        config=config,
        load_fn=lambda fp, **kw: load_bfcl_dataset(fp, test_category=config.test_category, **kw),
        description="BFCL v3 benchmark dataset loader",
    )
