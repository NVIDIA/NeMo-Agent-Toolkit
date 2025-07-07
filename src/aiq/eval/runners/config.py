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

import typing

from pydantic import BaseModel

from aiq.eval.config import EvaluationRunConfig
from aiq.eval.config import EvaluationRunOutput


class MultiEvaluationRunConfig(BaseModel):
    """
    Parameters used for a multi-evaluation run.
    This includes a base config and a dict of overrides. The key is an id of
    any type.
    Each pass loads the base config and runs to completion before the next pass
    starts.
    """
    base_config: EvaluationRunConfig
    endpoint: str | None = None
    endpoint_timeout: int = 300
    overrides: dict[typing.Any, tuple[tuple[str, str], ...]]


class MultiEvaluationRunOutput(BaseModel):
    """
    Output of a multi-evaluation run.
    The results per-pass are accumulated in the evaluation_runs dict.
    """
    evaluation_runs: dict[typing.Any, EvaluationRunOutput]
