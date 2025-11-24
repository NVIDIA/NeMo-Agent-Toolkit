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
Registration module for OpenPipe ART finetuning backend.

This module provides utilities for building and managing OpenPipe ART runners.
The actual registration is done in the finetuning_runtime module.
"""

import logging
from typing import AsyncIterator

from nat.finetuning.backends.openpipe_art.config import OpenPipeARTFinetuningConfig
from nat.finetuning.backends.openpipe_art.openpipe_art_runner import OpenPipeARTRunner

logger = logging.getLogger(__name__)


async def build_openpipe_art_runner(config: OpenPipeARTFinetuningConfig) -> AsyncIterator[OpenPipeARTRunner]:
    """
    Build and register the OpenPipe ART finetuning runner.
    
    Args:
        config: The finetuning configuration from config.yml
        
    Yields:
        OpenPipeARTRunner: The initialized finetuning runner
    """
    logger.info("Building OpenPipe ART finetuning runner")
    
    # Extract the runner config and trainer/run configs
    runner_config = config.runner_config
    
    # Use output_dir override if provided
    if config.output_dir:
        runner_config.output_dir = config.output_dir
    
    # Create the runner
    runner = OpenPipeARTRunner(
        trainer_config=runner_config.trainer_config,
        run_config=runner_config.run_config,
        num_generations=runner_config.num_generations
    )
    
    # Initialize the runner
    await runner.initialize()
    
    try:
        yield runner
    finally:
        # Cleanup when done
        await runner.cleanup()
