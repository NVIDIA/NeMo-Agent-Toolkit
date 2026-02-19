# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime data types for GA prompt optimization."""

from dataclasses import dataclass


@dataclass
class Individual:
    """
    One candidate solution in the GA population.

    - prompts: dotted path -> prompt text (one assignment across all prompt dimensions).
    - metrics: evaluator name -> average score; filled after evaluation.
    - scalar_fitness: single fitness value used for selection; set after normalize/scalarize/diversity.
    - worst_items_reasoning: optional reasoning strings from worst eval items for oracle feedback.
    """

    prompts: dict[str, str]
    metrics: dict[str, float] | None = None
    scalar_fitness: float | None = None
    worst_items_reasoning: list[str] | None = None
