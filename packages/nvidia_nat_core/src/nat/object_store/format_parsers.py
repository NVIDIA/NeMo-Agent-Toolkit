# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import io
import json
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import Any

_EXTENSION_TO_FORMAT: dict[str, str] = {
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "jsonl",
    ".parquet": "parquet",
    ".xls": "xls",
    ".xlsx": "xls",
}


def infer_format(key: str) -> str:
    """Infer data format from a key's file extension."""
    suffix = PurePosixPath(key).suffix.lower()
    if suffix not in _EXTENSION_TO_FORMAT:
        raise ValueError(f"Cannot infer format from key {key!r}. "
                         f"Supported extensions: {sorted(_EXTENSION_TO_FORMAT.keys())}. "
                         f"Specify format explicitly.")
    return _EXTENSION_TO_FORMAT[suffix]


def _get_parsers() -> dict[str, Callable[..., Any]]:
    """Lazily build the format parser registry. Avoids importing pandas at module level."""
    import pandas as pd

    return {
        "json": lambda data, **kw: pd.read_json(io.BytesIO(data), **kw),
        "jsonl": lambda data, **kw: pd.DataFrame([json.loads(line) for line in io.BytesIO(data) if line.strip()]),
        "csv": lambda data, **kw: pd.read_csv(io.BytesIO(data), **kw),
        "parquet": lambda data, **kw: pd.read_parquet(io.BytesIO(data), **kw),
        "xls": lambda data, **kw: pd.read_excel(io.BytesIO(data), engine="openpyxl", **kw),
    }


def parse_to_dataframe(data: bytes, format: str, **kwargs) -> Any:
    """Parse raw bytes into a pandas DataFrame using the specified format.

    Args:
        data: Raw bytes to parse.
        format: One of: csv, json, jsonl, parquet, xls.
        **kwargs: Forwarded to the underlying pandas reader.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If the format is unknown.
    """
    parsers = _get_parsers()
    if format not in parsers:
        raise ValueError(f"Unknown format: {format!r}. Supported: {sorted(parsers.keys())}")
    return parsers[format](data, **kwargs)
