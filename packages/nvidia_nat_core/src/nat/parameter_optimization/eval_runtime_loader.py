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

from functools import lru_cache


@lru_cache(maxsize=1)
def load_evaluation_run() -> type:
    """Lazily load eval runtime class required by `nat optimize`."""
    try:
        from nat.plugins.eval.runtime.evaluate import EvaluationRun
        return EvaluationRun
    except ImportError as exc:
        raise RuntimeError(
            "The `nat optimize` command requires evaluation support from `nvidia-nat-eval`. "
            "Install it with `uv pip install nvidia-nat-eval` (or `pip install nvidia-nat-eval`).") from exc
