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
"""Minimal custom evaluators for ATIF bridge example runs."""

from __future__ import annotations

from typing import Any


def artifact_presence_evaluator(atif_samples, artifact_path: str | None = None) -> dict[str, Any]:
    """Return reward `1.0` when at least one ATIF sample is present."""
    sample_count = len(atif_samples)
    reward = 1.0 if sample_count > 0 else 0.0
    return {
        "reward": reward,
        "details": {
            "sample_count": sample_count,
            "artifact_path": artifact_path,
        },
    }
