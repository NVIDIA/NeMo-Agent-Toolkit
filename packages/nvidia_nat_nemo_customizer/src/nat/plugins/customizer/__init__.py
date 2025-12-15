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
NeMo Customizer plugin for NAT finetuning.

This plugin provides trajectory builders and trainer adapters for
finetuning workflows using NeMo Customizer backend.

Available components:
- DPO Trajectory Builder: Collects preference pairs from scored TTC candidates
- NeMo Customizer TrainerAdapter: Submits DPO/SFT jobs to NeMo Customizer
"""

from .dpo import DPOSpecificHyperparameters
from .dpo import DPOTrajectoryBuilder
from .dpo import DPOTrajectoryBuilderConfig
from .dpo import NeMoCustomizerHyperparameters
from .dpo import NeMoCustomizerTrainerAdapter
from .dpo import NeMoCustomizerTrainerAdapterConfig
from .dpo import NIMDeploymentConfig

__all__ = [
    # Trajectory Builder
    "DPOTrajectoryBuilder",
    "DPOTrajectoryBuilderConfig",  # TrainerAdapter
    "NeMoCustomizerTrainerAdapter",
    "NeMoCustomizerTrainerAdapterConfig",
    "NeMoCustomizerHyperparameters",
    "DPOSpecificHyperparameters",
    "NIMDeploymentConfig",
]
