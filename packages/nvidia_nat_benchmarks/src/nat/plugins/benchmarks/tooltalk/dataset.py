# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path

import pandas as pd

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader

from .config import ToolTalkDatasetConfig

logger = logging.getLogger(__name__)


def load_tooltalk_dataset(file_path: str | Path) -> pd.DataFrame:
    """Load a ToolTalk dataset directory into a DataFrame.

    Each JSON file in the directory becomes one row. The full conversation JSON
    is serialized into the 'question' column so the workflow can deserialize it.
    The ground truth conversation (with expected API calls) goes into 'answer'.
    """
    data_dir = Path(file_path)
    if not data_dir.is_dir():
        raise ValueError(f"ToolTalk dataset path must be a directory, got: {file_path}")

    rows = []
    for json_file in sorted(data_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as f:
            conversation = json.load(f)

        rows.append({
            "id": conversation.get("conversation_id", json_file.stem),
            "question": json.dumps(conversation),
            "answer": json.dumps(conversation),
        })

    if not rows:
        raise ValueError(f"No JSON files found in ToolTalk dataset directory: {file_path}")

    logger.info("Loaded %d ToolTalk conversations from %s", len(rows), data_dir)
    return pd.DataFrame(rows)


@register_dataset_loader(config_type=ToolTalkDatasetConfig)
async def register_tooltalk_dataset_loader(config: ToolTalkDatasetConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(
        config=config,
        load_fn=load_tooltalk_dataset,
        description="ToolTalk benchmark dataset loader (directory of JSON conversations)",
    )
