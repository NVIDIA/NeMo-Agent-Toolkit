# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

"""Finetuning runtime for NAT that orchestrates the training process."""

import asyncio
import logging
from pathlib import Path

from nat.data_models.config import Config

logger = logging.getLogger(__name__)


async def run_finetuning(config: Config) -> None:
    """
    Run finetuning based on the provided configuration.

    Args:
        config: The NAT configuration containing finetuning settings
    """
    pass


async def finetuning_main(config_file: Path) -> None:
    """
    Main entry point for finetuning runtime.

    Args:
        config_file: Path to the configuration file
    """

    from nat.runtime.loader import load_config
    config = load_config(config_file=config_file)

    # Run finetuning
    await run_finetuning(config)


def run_finetuning_sync(config_file: Path) -> None:
    """
    Synchronous wrapper for running finetuning.

    Args:
        config_file: Path to the configuration file
    """
    asyncio.run(finetuning_main(config_file))
