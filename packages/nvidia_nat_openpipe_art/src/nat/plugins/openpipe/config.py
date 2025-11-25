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

from pydantic import Field

from nat.data_models.finetuning import TrajectoryBuilderConfig


class OpenPipeARTTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="openpipe_art_traj_builder"):
    """
    Configuration for the OpenPipe ART Trajectory Builder.
    """
    num_generations: int = Field(default=2,
                                 description="Number of trajectory generations per example in eval dataset",
                                 ge=1)
