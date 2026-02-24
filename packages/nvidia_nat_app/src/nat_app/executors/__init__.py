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
"""
Runtime execution primitives.

Execution runner:
    - ``run_speculation`` -- core speculation lifecycle (launch,
      await, decide, cancel, collect).
    - ``SpeculativeResult`` -- outcome dataclass returned by the runner.

Runtime utilities:
    - ``ExecutionState`` -- node lifecycle state machine.
    - ``ExecutionMetrics`` -- standardized execution metrics.
    - ``ResultHandler`` -- pluggable result type dispatch.

Speculation planning and safety primitives live in ``nat_app.speculation``.
"""

from nat_app.executors.execution_state import ExecutionState
from nat_app.executors.metrics import ExecutionMetrics
from nat_app.executors.result_handler import ResultHandler
from nat_app.executors.runner import SpeculativeResult
from nat_app.executors.runner import run_speculation
from nat_app.speculation import RouterDescriptor
from nat_app.speculation import SpeculationPlan
from nat_app.speculation import SpeculationSafetyConfig
from nat_app.speculation import is_marked_speculation_unsafe
from nat_app.speculation import partition_targets
from nat_app.speculation import speculation_unsafe

__all__ = [
    "ExecutionMetrics",
    "ExecutionState",
    "ResultHandler",
    "RouterDescriptor",
    "SpeculationPlan",
    "SpeculationSafetyConfig",
    "SpeculativeResult",
    "is_marked_speculation_unsafe",
    "partition_targets",
    "run_speculation",
    "speculation_unsafe",
]
