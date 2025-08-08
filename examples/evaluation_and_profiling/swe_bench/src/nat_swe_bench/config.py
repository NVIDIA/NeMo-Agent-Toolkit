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

from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag

from nat.data_models.common import BaseModelRegistryTag
from nat.data_models.common import TypedBaseModel
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class SweBenchPredictorBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    description: str = "Swe Bench Problem Solver"


class SweBenchPredictorFullConfig(SweBenchPredictorBaseConfig, name="full"):
    llm_name: LLMRef = "nim_llm"
    tool_names: list[FunctionRef] = []
    # Temporary, key needs to be removed and read from the environment
    openai_api_key: str = Field(default="")  # OpenAI API key field


class SweBenchPredictorGoldConfig(SweBenchPredictorBaseConfig, name="gold"):
    verbose: bool = True


class SweBenchPredictorSkeletonConfig(SweBenchPredictorBaseConfig, name="skeleton"):
    verbose: bool = False


SweBenchPredictorConfig = typing.Annotated[
    typing.Annotated[SweBenchPredictorFullConfig, Tag(SweBenchPredictorFullConfig.static_type())]
    | typing.Annotated[SweBenchPredictorGoldConfig, Tag(SweBenchPredictorGoldConfig.static_type())]
    | typing.Annotated[SweBenchPredictorSkeletonConfig, Tag(SweBenchPredictorSkeletonConfig.static_type())],
    Discriminator(TypedBaseModel.discriminator)]


class SweBenchWorkflowConfig(FunctionBaseConfig, name="swe_bench"):
    predictor: SweBenchPredictorConfig
