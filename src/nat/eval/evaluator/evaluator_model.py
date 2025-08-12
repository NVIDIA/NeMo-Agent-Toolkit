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

from nat.data_models.intermediate_step import IntermediateStep


class EvalInputItem(BaseModel):
    id: typing.Any
    input_obj: typing.Any
    expected_output_obj: typing.Any
    output_obj: typing.Any
    expected_trajectory: list[IntermediateStep]
    trajectory: list[IntermediateStep]
    full_dataset_entry: typing.Any


class EvalInput(BaseModel):
    eval_input_items: list[EvalInputItem]


class EvalOutputItem(BaseModel):
    id: typing.Any  # id or input_obj from EvalInputItem
    score: typing.Any  # float or any serializable type
    reasoning: typing.Any


class EvalOutput(BaseModel):
    average_score: typing.Any  # float or any serializable type
    eval_output_items: list[EvalOutputItem]
