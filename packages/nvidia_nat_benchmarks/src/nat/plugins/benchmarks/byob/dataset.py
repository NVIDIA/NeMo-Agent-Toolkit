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
"""BYOB dataset loader.

Imports a BYOB benchmark definition and loads its dataset into a DataFrame
compatible with NAT's eval runner.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from nat.builder.builder import EvalBuilder
from nat.builder.dataset_loader import DatasetLoaderInfo
from nat.cli.register_workflow import register_dataset_loader

from .config import BYOBDatasetConfig

logger = logging.getLogger(__name__)


def load_byob_dataset(
    file_path: str | Path,
    benchmark_module: str = "",
    benchmark_name: str = "",
    limit: int | None = None,
) -> pd.DataFrame:
    """Load a BYOB benchmark's dataset into a DataFrame.

    Uses import_benchmark() to resolve the benchmark definition, then loads
    the dataset using BYOB's dataset loading utilities.

    The file_path parameter is required by NAT's DatasetHandler interface but
    is ignored — the dataset path comes from the benchmark definition.
    """
    from nemo_evaluator.contrib.byob.eval_logic import import_benchmark
    from nemo_evaluator.contrib.byob.dataset import load_dataset

    bench = import_benchmark(benchmark_module, benchmark_name)
    logger.info("Imported BYOB benchmark '%s' (dataset: %s)", bench.name, bench.dataset)

    # Load the raw dataset
    samples = load_dataset(bench.dataset, limit=limit, field_mapping=bench.field_mapping)

    rows = []
    for i, sample in enumerate(samples):
        target = sample.get(bench.target_field, "")
        # Use response_field for eval-only mode if present
        response = sample.get(bench.response_field, "") if bench.response_field else ""

        rows.append({
            "id": sample.get("id", str(i)),
            "question": json.dumps(sample),
            "answer": json.dumps(target) if not isinstance(target, str) else target,
        })

    logger.info("Loaded %d BYOB samples from benchmark '%s'", len(rows), bench.name)
    return pd.DataFrame(rows)


@register_dataset_loader(config_type=BYOBDatasetConfig)
async def register_byob_dataset_loader(config: BYOBDatasetConfig, builder: EvalBuilder):
    yield DatasetLoaderInfo(
        config=config,
        load_fn=lambda fp, **kw: load_byob_dataset(
            fp, benchmark_module=config.benchmark_module,
            benchmark_name=config.benchmark_name, limit=config.limit, **kw,
        ),
        description="BYOB benchmark dataset loader",
    )
