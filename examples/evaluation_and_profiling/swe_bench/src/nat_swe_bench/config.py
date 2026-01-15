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

import typing

from pydantic import Discriminator
from pydantic import Field
from pydantic import Tag

from nat.data_models.common import BaseModelRegistryTag
from nat.data_models.common import TypedBaseModel
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


class SweBenchPredictorBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration class for SWE-bench predictors."""    
    description: str = "Swe Bench Problem Solver"


class SweBenchPredictorGoldConfig(SweBenchPredictorBaseConfig, name="gold"):
    """Configuration for the gold predictor that uses the provided patch directly.
    
    Attributes:
        verbose: Whether to enable verbose output for debugging.
    """
    verbose: bool = True


class SweBenchPredictorSkeletonConfig(SweBenchPredictorBaseConfig, name="skeleton"):
    """Configuration for the skeleton predictor template.
    
    Attributes:
        verbose: Whether to enable verbose output for debugging.
    """
    verbose: bool = False

class SweBenchPredictorIterativeConfig(SweBenchPredictorBaseConfig, name="iterative"):
    """Configuration for the iterative predictor that solves problems step-by-step.
    
    Attributes:
        llm_name: Reference to the LLM to use for iterative problem solving.
        step_limit: Maximum number of agent steps before termination.
        timeout: Command execution timeout in seconds.
    """    
    llm_name: LLMRef = Field(description="LLM to use for iterative agent")
    step_limit: int = Field(default=250, description="Maximum number of agent steps")
    timeout: int = Field(default=60, description="Command execution timeout in seconds")

SweBenchPredictorConfig = typing.Annotated[
    typing.Annotated[SweBenchPredictorGoldConfig, Tag(SweBenchPredictorGoldConfig.static_type())]
    | typing.Annotated[SweBenchPredictorSkeletonConfig, Tag(SweBenchPredictorSkeletonConfig.static_type())]
    | typing.Annotated[SweBenchPredictorIterativeConfig, Tag(SweBenchPredictorIterativeConfig.static_type())],
    Discriminator(TypedBaseModel.discriminator)]

class SweBenchWorkflowConfig(FunctionBaseConfig, name="swe_bench"):
    """Configuration for the SWE-bench workflow.
    
    Attributes:
        predictor: The predictor configuration (gold, skeleton, or iterative).
    """
    predictor: SweBenchPredictorConfig
