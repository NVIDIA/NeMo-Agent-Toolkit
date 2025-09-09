# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

from nat.data_models.intermediate_step import IntermediateStep
from nat.profiler.inference_optimization.data_models import WorkflowRuntimeMetrics
from nat.profiler.utils import create_standardized_dataframe


def compute_workflow_runtime_metrics(all_steps: list[list[IntermediateStep]]) -> WorkflowRuntimeMetrics:
    """
    Computes the p90, p95, and p99 of workflow runtime for each example_number.

    The 'workflow runtime' per example is::

        max(event_timestamp) - min(event_timestamp)

    for that example_number.

    Parameters
    ----------
    all_steps : IntermediateStep
        Must contain at least two columns:
          - 'example_number'
          - 'event_timestamp'

    Returns
    -------
    WorkflowRuntimeMetrics
        A Pydantic model with 'p90', 'p95', and 'p99' attributes.
    """
    df = create_standardized_dataframe(all_steps)
    required_cols = {"example_number", "event_timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Group by example_number, then find min and max timestamp
    grouped = df.groupby("example_number")["event_timestamp"]
    min_timestamps = grouped.min()
    max_timestamps = grouped.max()

    # Workflow runtime is difference between max and min
    runtimes = max_timestamps - min_timestamps

    # Convert to a NumPy array for percentile calculations
    runtimes_arr = runtimes.values

    # Edge case: if there's only one example or no data
    # (NumPy percentile can handle 1-element arrays, but let's guard for empties)
    if len(runtimes_arr) == 0:
        return WorkflowRuntimeMetrics(p90=0.0, p95=0.0, p99=0.0)

    # Compute p90, p95, p99
    p90_val = float(np.percentile(runtimes_arr, 90))
    p95_val = float(np.percentile(runtimes_arr, 95))
    p99_val = float(np.percentile(runtimes_arr, 99))

    return WorkflowRuntimeMetrics(p90=p90_val, p95=p95_val, p99=p99_val)
