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
from pydantic import ConfigDict
from pydantic import Field


class ITSItem(BaseModel):
    """
    Represents an item in the Inference Time Scaling (ITS) functions and pipelines
    """
    model_config = ConfigDict(extra="allow")

    input: typing.Any | None = Field(default=None,
                                     description="Input to the function or pipeline. "
                                     "This can be a structured tool call, or other info.")
    output: typing.Any | None = Field(default=None,
                                      description="Output from the function or pipeline. "
                                      "This can be a structured tool call, or other info.")
    plan: typing.Any | None = Field(default=None, description="Search plan for downstream agent(s).")
    feedback: str | None = Field(default=None,
                                 description="Feedback "
                                 "provided by feedback steps to improve the plan.")
    score: float | None = Field(default=None,
                                description="Score of the plan based on feedback or other evaluation criteria. "
                                "This can be used to rank plans.")
    metadata: typing.Any | None = Field(default=None,
                                        description="Additional information. This can be"
                                        " a structured tool call, or other info not "
                                        "in the plan.")
    name: str | None = Field(default=None,
                             description="Name of the item or function"
                             ", used for identification in pipelines.")
