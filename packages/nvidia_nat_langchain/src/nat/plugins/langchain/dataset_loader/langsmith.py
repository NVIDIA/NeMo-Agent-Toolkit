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

import pandas as pd
from langsmith import Client


def load_langsmith_dataset(
    file_path,
    *,
    dataset_id: str | None = None,
    dataset_name: str | None = None,
    input_key: str = "input",
    output_key: str = "output",
    question_col: str = "question",
    answer_col: str = "answer",
    id_col: str = "id",
    split: str | None = None,
    as_of: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Fetch a dataset from LangSmith and return as a pandas DataFrame.

    Prefers dataset_id over dataset_name when both are provided.
    The file_path argument is ignored — data comes from the LangSmith API, not the filesystem.

    Loads dataset of format https://docs.langchain.com/langsmith/example-data-format
    """
    client = Client()  # reads LANGCHAIN_API_KEY / LANGSMITH_API_KEY from env

    # Prefer dataset_id over dataset_name
    list_kwargs: dict = {}
    if dataset_id:
        list_kwargs["dataset_id"] = dataset_id
    elif dataset_name:
        list_kwargs["dataset_name"] = dataset_name
    else:
        raise ValueError("At least one of 'dataset_id' or 'dataset_name' must be provided")

    if split:
        list_kwargs["splits"] = [split]
    if as_of:
        list_kwargs["as_of"] = as_of

    rows: list[dict] = []
    for i, ex in enumerate(client.list_examples(**list_kwargs)):
        if limit is not None and i >= limit:
            break
        row = {
            id_col: str(ex.id),
            question_col: ex.inputs.get(input_key, ""),
            answer_col: (ex.outputs or {}).get(output_key, ""),
        }
        # Include all original fields for full_dataset_entry
        for k, v in ex.inputs.items():
            if k not in row:
                row[k] = v
        if ex.outputs:
            for k, v in ex.outputs.items():
                if k not in row:
                    row[k] = v
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[id_col, question_col, answer_col])

    return pd.DataFrame(rows)
