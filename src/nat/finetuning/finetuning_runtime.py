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

from nat.data_models.finetuning import FinetuneRunConfig
from nat.finetuning.interfaces.finetuning_runner import Trainer

logger = logging.getLogger(__name__)


async def run_finetuning(runner: Trainer) -> None:
    """
    Run finetuning based on the provided configuration.

    Args:
        runner: An instance of the Trainer to run finetuning with
    """
    try:
        # Initialize the runner
        logger.info("Initializing finetuning runner...")

        # Get number of epochs from config
        num_epochs = runner.run_config.num_epochs

        # Run training for specified epochs
        logger.info("Starting training for %d epochs...", num_epochs)
        job_statuses = await runner.run(num_epochs)

        # Log final status
        for status in job_statuses:
            logger.info("Job %s completed with status: %s", status.run_id, status.status)
            if status.message:
                logger.info("  Message: %s", status.message)

        # Get and log final metrics
        if job_statuses:
            final_run_id = job_statuses[-1].run_id
            try:
                metrics = await runner.get_metrics(final_run_id)
                logger.info("Final metrics: %s", metrics)
            except (ValueError, RuntimeError) as e:
                logger.warning("Failed to retrieve metrics: %s", e)

        logger.info("Finetuning completed successfully!")

    except Exception as e:
        logger.error("Finetuning failed: %s", e)
        raise
    finally:
        # Always cleanup resources
        logger.info("Cleaning up finetuning resources...")
        await runner.cleanup()
        logger.info("Cleanup completed")


async def finetuning_main(run_config: FinetuneRunConfig) -> None:
    """
    Main entry point for finetuning runtime.

    Args:
        run_config: FinetuneRunConfig object containing finetuning settings
    """

    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config

    config = load_config(config_file=run_config.config_file)
    finetuning_config = config.finetuning
    finetuning_config.run_configuration = run_config

    if not config.finetuning.enabled:
        raise ValueError("Finetuning is not enabled in the provided configuration.")

    async with WorkflowBuilder.from_config(config=config) as builder:
        # Get trajectory builder and trainer adapter from builder
        logger.info("Initializing finetuning components...")
        trajectory_builder_name = finetuning_config.trajectory_builder
        trainer_adapter_name = finetuning_config.trainer_adapter
        trajectory_builder = await builder.get_trajectory_builder(trajectory_builder_name)
        trainer_adapter = await builder.get_trainer_adapter(trainer_adapter_name)
        logger.info("Finetuning components initialized.")

        # Initialize trainer
        trainer_name = finetuning_config.trainer
        trainer = await builder.get_trainer(trainer_name,
                                            trajectory_builder=trajectory_builder,
                                            trainer_adapter=trainer_adapter)

        await trainer.initialize(run_config=finetuning_config)

        logger.info("Initialized trainer: %s", trainer_name)

    # Run finetuning
    await run_finetuning(trainer)


def run_finetuning_sync(run_config: FinetuneRunConfig) -> None:
    """
    Synchronous wrapper for running finetuning.

    Args:
        run_config: FinetuneRunConfig object containing finetuning settings
    """
    asyncio.run(finetuning_main(run_config))
