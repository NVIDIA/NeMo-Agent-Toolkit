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

from nat.eval.config import EvaluationRunConfig
from nat.eval.config import EvaluationRunOutput


class MultiEvaluationRunConfig(BaseModel):
    """
    Parameters used for a multi-evaluation run.
    This includes a dict of configs. The key is an id of any type.
    Each pass loads the config, applies the overrides and runs to completion
    before the next pass starts.
    """
    configs: dict[typing.Any, EvaluationRunConfig]


class MultiEvaluationRunOutput(BaseModel):
    """
    Output of a multi-evaluation run.
    The results per-pass are accumulated in the evaluation_run_outputs dict.
    """
    evaluation_run_outputs: dict[typing.Any, EvaluationRunOutput]
