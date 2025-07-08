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
"""
This file contains the abstract base class for SWE Bench predictor workflows.
"""

from abc import ABC
from abc import abstractmethod

from aiq.builder.builder import Builder
from aiq.data_models.swe_bench_model import SWEBenchInput

from ..config import SweBenchWorkflowConfig


class SweBenchPredictorBase(ABC):
    """
    Abstract Base Class for SWE Bench workflow functions.
    """

    def __init__(self, config: SweBenchWorkflowConfig, builder: Builder):
        """
        Initialize the workflow with configuration and builder.

        Args:
            config (SweBenchWorkflowConfig): Workflow configuration object.
            builder (Builder): Workflow builder for setup.
        """
        self.config = config
        self.builder = builder

    @abstractmethod
    async def predict_fn(self, swebench_input: SWEBenchInput) -> str:
        """
        Predict function to generate or fetch a patch for the SWE Bench input.

        Args:
            swebench_input (SWEBenchInput): Input data for SWE Bench instance.

        Returns:
            str: The predicted patch string.
        """
        pass
