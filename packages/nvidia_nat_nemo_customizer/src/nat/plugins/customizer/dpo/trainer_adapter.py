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
NeMo Customizer TrainerAdapter for DPO/SFT training.

This module provides a TrainerAdapter implementation that interfaces with
NeMo Customizer for submitting and monitoring training jobs.
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import HfApi
from nemo_microservices import NeMoMicroservices

from nat.data_models.finetuning import DPOItem
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import TrajectoryCollection
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter

from .config import NeMoCustomizerTrainerAdapterConfig

logger = logging.getLogger(__name__)


class NeMoCustomizerTrainerAdapter(TrainerAdapter):
    """
    TrainerAdapter for NeMo Customizer backend.

    This adapter:
    1. Converts trajectories to JSONL format for DPO training
    2. Uploads datasets to NeMo Datastore via HuggingFace Hub API
    3. Submits customization jobs to NeMo Customizer
    4. Monitors job progress and status
    5. Optionally deploys trained models
    """

    def __init__(self, adapter_config: NeMoCustomizerTrainerAdapterConfig):
        super().__init__(adapter_config)

        self.adapter_config: NeMoCustomizerTrainerAdapterConfig = adapter_config

        # Initialize NeMo Microservices client
        self._entity_client: NeMoMicroservices | None = None
        self._hf_api: HfApi | None = None

        # Track active jobs
        self._active_jobs: dict[str, str] = {}  # run_id -> job_id mapping
        self._job_output_models: dict[str, str] = {}  # run_id -> output_model mapping

        logger.info(f"Initialized NeMoCustomizerTrainerAdapter for namespace: {adapter_config.namespace}")

    @property
    def entity_client(self) -> NeMoMicroservices:
        """Lazy initialization of NeMo Microservices client."""
        if self._entity_client is None:
            self._entity_client = NeMoMicroservices(base_url=self.adapter_config.entity_host)
        return self._entity_client

    @property
    def hf_api(self) -> HfApi:
        """Lazy initialization of HuggingFace API client."""
        if self._hf_api is None:
            self._hf_api = HfApi(
                endpoint=f"{self.adapter_config.datastore_host}/v1/hf",
                token=self.adapter_config.hf_token or "",
            )
        return self._hf_api

    async def initialize(self, run_config: FinetuneConfig) -> None:
        """Initialize the trainer adapter."""
        await super().initialize(run_config)

        if self.adapter_config.create_namespace_if_missing:
            await self._ensure_namespaces_exist()

        health = await self.is_healthy()
        if not health:
            raise ConnectionError(f"Failed to connect to NeMo Customizer at {self.adapter_config.entity_host}")

        logger.info("Successfully initialized NeMo Customizer TrainerAdapter")

    async def _ensure_namespaces_exist(self) -> None:
        """Create namespaces in entity store and datastore if they don't exist."""
        namespace = self.adapter_config.namespace

        # Create namespace in entity store
        try:
            self.entity_client.namespaces.create(
                id=namespace,
                description=f"NAT finetuning namespace: {namespace}",
            )
            logger.info(f"Created namespace '{namespace}' in Entity Store")
        except Exception as e:
            logger.debug(f"Namespace '{namespace}' may already exist in Entity Store: {e}")

        # Create namespace in datastore via HTTP
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.adapter_config.datastore_host}/v1/datastore/namespaces",
                    data={"namespace": namespace},
                )
                if resp.status_code in (200, 201):
                    logger.info(f"Created namespace '{namespace}' in Datastore")
                elif resp.status_code in (409, 422):
                    logger.debug(f"Namespace '{namespace}' already exists in Datastore")
                else:
                    logger.warning(f"Unexpected response creating namespace in Datastore: {resp.status_code}")
        except Exception as e:
            logger.warning(f"Error creating namespace in Datastore: {e}")

    async def is_healthy(self) -> bool:
        """Check if NeMo Customizer services are reachable."""
        return True

    def _format_prompt(self, prompt: list[OpenAIMessage] | str) -> list[dict[str, str]] | str:
        """
        Format prompt based on configuration.

        Args:
            prompt: Original prompt (string or list of OpenAI messages)

        Returns:
            Formatted prompt based on use_full_message_history setting
        """
        if self.adapter_config.use_full_message_history:
            # Return full message history as list of dicts
            if isinstance(prompt, str):
                return [{"role": "user", "content": prompt}]
            else:
                return [{"role": msg.role, "content": msg.content} for msg in prompt]
        # Return only last message content as string
        elif isinstance(prompt, str):
            return prompt
        elif prompt:
            return prompt[-1].content
        else:
            return ""

    def _trajectory_to_dpo_jsonl(self, trajectories: TrajectoryCollection) -> tuple[str, str]:
        """
        Convert trajectory collection to JSONL format for DPO training.

        Returns:
            Tuple of (training_jsonl, validation_jsonl) content strings
        """
        all_items: list[dict[str, Any]] = []

        for trajectory_group in trajectories.trajectories:
            for trajectory in trajectory_group:
                for episode_item in trajectory.episode:
                    if isinstance(episode_item, DPOItem):
                        formatted_prompt = self._format_prompt(episode_item.prompt)
                        dpo_record = {
                            "prompt": formatted_prompt,
                            "chosen_response": episode_item.chosen_response,
                            "rejected_response": episode_item.rejected_response,
                        }
                        all_items.append(dpo_record)

        if not all_items:
            raise ValueError("No DPO items found in trajectories")

        # Split into training (80%) and validation (20%)
        split_idx = max(1, int(len(all_items) * 0.8))
        training_items = all_items[:split_idx]
        validation_items = all_items[split_idx:] if split_idx < len(all_items) else all_items[-1:]

        training_jsonl = "\n".join(json.dumps(item) for item in training_items)
        validation_jsonl = "\n".join(json.dumps(item) for item in validation_items)

        logger.info(f"Converted {len(all_items)} DPO items: "
                    f"{len(training_items)} training, {len(validation_items)} validation")

        return training_jsonl, validation_jsonl

    async def _setup_dataset(self, run_id: str, training_jsonl: str, validation_jsonl: str) -> str:
        """
        Create dataset repository and upload JSONL files.

        Args:
            run_id: Unique identifier for this training run
            training_jsonl: Training data in JSONL format
            validation_jsonl: Validation data in JSONL format

        Returns:
            Repository ID for the created dataset
        """
        dataset_name = f"{self.adapter_config.dataset_name}"
        repo_id = f"{self.adapter_config.namespace}/{dataset_name}"

        # Create dataset repo in datastore
        self.hf_api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

        # Register dataset in entity store
        try:
            self.entity_client.datasets.create(
                name=dataset_name,
                namespace=self.adapter_config.namespace,
                files_url=f"hf://datasets/{repo_id}",
                description=f"NAT DPO training dataset for run {run_id}",
            )
        except Exception as e:
            logger.debug(f"Dataset may already exist: {e}")

        # Determine output directory for dataset files
        if self.adapter_config.dataset_output_dir:
            # Use configured output directory (create if needed, preserve files)
            output_dir = Path(self.adapter_config.dataset_output_dir) / run_id
            output_dir.mkdir(parents=True, exist_ok=True)
            use_temp_dir = False
            logger.info(f"Saving dataset files to: {output_dir}")
        else:
            # Use temporary directory (will be cleaned up)
            use_temp_dir = True

        def write_and_upload_files(base_dir: Path) -> None:
            train_path = base_dir / "training_file.jsonl"
            val_path = base_dir / "validation_file.jsonl"

            train_path.write_text(training_jsonl)
            val_path.write_text(validation_jsonl)

            self.hf_api.upload_file(
                path_or_fileobj=str(train_path),
                path_in_repo="training/training_file.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                revision="main",
                commit_message=f"Training file for run {run_id}",
            )

            self.hf_api.upload_file(
                path_or_fileobj=str(val_path),
                path_in_repo="validation/validation_file.jsonl",
                repo_id=repo_id,
                repo_type="dataset",
                revision="main",
                commit_message=f"Validation file for run {run_id}",
            )

        if use_temp_dir:
            with tempfile.TemporaryDirectory() as tmpdir:
                write_and_upload_files(Path(tmpdir))
        else:
            write_and_upload_files(output_dir)

        logger.info(f"Created and uploaded dataset: {repo_id}")
        return dataset_name

    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """
        Submit trajectories for training.

        Args:
            trajectories: Collection of trajectories containing DPO items

        Returns:
            Reference to the submitted training job
        """
        run_id = trajectories.run_id

        if run_id in self._active_jobs:
            raise ValueError(f"Training job for run {run_id} already exists")

        # Convert trajectories to JSONL
        training_jsonl, validation_jsonl = self._trajectory_to_dpo_jsonl(trajectories)

        # Upload dataset
        dataset_name = await self._setup_dataset(run_id, training_jsonl, validation_jsonl)

        # Prepare hyperparameters
        hyperparams = self.adapter_config.hyperparameters.model_dump()

        # Submit customization job
        job = self.entity_client.customization.jobs.create(
            config=self.adapter_config.customization_config,
            dataset={
                "name": dataset_name,
                "namespace": self.adapter_config.namespace,
            },
            hyperparameters=hyperparams,
        )

        job_id = job.id
        self._active_jobs[run_id] = job_id
        self._job_output_models[run_id] = job.output_model

        logger.info(f"Submitted customization job {job_id} for run {run_id}. "
                    f"Output model: {job.output_model}")

        return TrainingJobRef(
            run_id=run_id,
            backend="nemo-customizer",
            metadata={
                "job_id": job_id,
                "output_model": job.output_model,
                "dataset_name": dataset_name,
            },
        )

    async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
        """Get the status of a training job."""
        job_id = self._active_jobs.get(ref.run_id)
        if job_id is None:
            # Try to get from metadata
            job_id = ref.metadata.get("job_id") if ref.metadata else None

        if job_id is None:
            raise ValueError(f"No training job found for run {ref.run_id}")

        try:
            job_status = self.entity_client.customization.jobs.status(job_id)

            # Map NeMo status to TrainingStatusEnum
            status_map = {
                "created": TrainingStatusEnum.PENDING,
                "pending": TrainingStatusEnum.PENDING,
                "running": TrainingStatusEnum.RUNNING,
                "completed": TrainingStatusEnum.COMPLETED,
                "failed": TrainingStatusEnum.FAILED,
                "cancelled": TrainingStatusEnum.CANCELED,
                "canceled": TrainingStatusEnum.CANCELED,
            }

            status = status_map.get(job_status.status.lower(), TrainingStatusEnum.RUNNING)
            progress = getattr(job_status, "percentage_done", None)

            message = f"Status: {job_status.status}"
            if hasattr(job_status, "epochs_completed"):
                message += f", Epochs: {job_status.epochs_completed}"

            return TrainingJobStatus(
                run_id=ref.run_id,
                backend=ref.backend,
                status=status,
                progress=progress,
                message=message,
                metadata={
                    "job_id": job_id,
                    "nemo_status": job_status.status,
                    "output_model": self._job_output_models.get(ref.run_id),
                },
            )

        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return TrainingJobStatus(
                run_id=ref.run_id,
                backend=ref.backend,
                status=TrainingStatusEnum.FAILED,
                message=f"Error getting status: {e}",
            )

    async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float | None = None) -> TrainingJobStatus:
        """Wait for training job to complete."""
        interval = poll_interval or self.adapter_config.poll_interval_seconds

        last_status: str | None = None

        while True:
            status = await self.status(ref)

            # Log when status changes
            current_status = status.status.value
            if current_status != last_status:
                logger.info(f"Job {ref.run_id}: Status -> '{current_status}'")
                last_status = current_status

            # Log when progress changes
            current_progress = status.progress
            #if current_progress is not None and current_progress != last_progress:
            logger.info(f"Job {ref.run_id}: Progress {current_progress:.1f}%")

            if status.status in (
                    TrainingStatusEnum.COMPLETED,
                    TrainingStatusEnum.FAILED,
                    TrainingStatusEnum.CANCELED,
            ):
                # Handle deployment if configured
                if (status.status == TrainingStatusEnum.COMPLETED and self.adapter_config.deploy_on_completion):
                    await self._deploy_model(ref)

                # Clean up active job tracking
                self._active_jobs.pop(ref.run_id, None)

                return status

            await asyncio.sleep(interval)

    async def _deploy_model(self, ref: TrainingJobRef) -> None:
        """Deploy the trained model and wait until deployment is ready."""
        output_model = self._job_output_models.get(ref.run_id)
        if not output_model:
            logger.warning(f"No output model found for run {ref.run_id}, skipping deployment")
            return

        deploy_config = self.adapter_config.deployment_config
        namespace = self.adapter_config.namespace

        try:
            # Create deployment configuration
            config_name = f"nat-deploy-config-{ref.run_id}"
            dep_config = self.entity_client.deployment.configs.create(
                name=config_name,
                namespace=namespace,
                description=deploy_config.description,
                model=output_model,
                nim_deployment={
                    "image_name": deploy_config.image_name,
                    "image_tag": deploy_config.image_tag,
                    "gpu": deploy_config.gpu,
                },
            )

            # Create model deployment
            deployment_name = (deploy_config.deployment_name or f"nat-deployment-{ref.run_id}")
            self.entity_client.deployment.model_deployments.create(
                name=deployment_name,
                namespace=namespace,
                description=deploy_config.description,
                config=f"{dep_config.namespace}/{dep_config.name}",
            )

            logger.info(f"Created deployment '{deployment_name}' for model {output_model}")

            # Wait for deployment to be ready
            await self._wait_for_deployment_ready(namespace, deployment_name)

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise

    async def _wait_for_deployment_ready(
        self,
        namespace: str,
        deployment_name: str,
        poll_interval: float | None = None,
        timeout: float | None = None,
    ) -> None:
        """
        Wait for a model deployment to become ready.

        Args:
            namespace: Namespace of the deployment
            deployment_name: Name of the deployment
            poll_interval: Seconds between status checks (default: adapter config poll_interval_seconds)
            timeout: Maximum seconds to wait (default: adapter config deployment_timeout_seconds)
        """
        interval = poll_interval or self.adapter_config.poll_interval_seconds
        max_wait = timeout or self.adapter_config.deployment_timeout_seconds

        logger.info(f"Waiting for deployment '{deployment_name}' to be ready...")

        last_status: str | None = None
        elapsed = 0.0

        while elapsed < max_wait:
            try:
                # Get all deployments and find ours
                deployments = self.entity_client.deployment.model_deployments.list().data
                deployment = None
                for dep in deployments:
                    if dep.name == deployment_name and dep.namespace == namespace:
                        deployment = dep
                        break

                if deployment is None:
                    logger.warning(f"Deployment '{deployment_name}' not found in namespace '{namespace}'")
                    await asyncio.sleep(interval)
                    elapsed += interval
                    continue

                # Check status
                status_details = getattr(deployment, "status_details", None)
                current_status = status_details.status if status_details else "unknown"
                description = status_details.description if status_details else ""

                # Log status changes
                if current_status != last_status:
                    logger.info(f"Deployment '{deployment_name}': Status -> '{current_status}'")
                    if description:
                        logger.info(f"Deployment '{deployment_name}': {description.strip()}")
                    last_status = current_status

                # Check if ready
                if current_status.lower() == "ready":
                    logger.info(f"Deployment '{deployment_name}' is ready!")
                    return

                # Check for failure states
                if current_status.lower() in ("failed", "error"):
                    raise RuntimeError(
                        f"Deployment '{deployment_name}' failed with status '{current_status}': {description}")

            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"Error checking deployment status: {e}")

            await asyncio.sleep(interval)
            elapsed += interval

        raise TimeoutError(f"Deployment '{deployment_name}' did not become ready within {max_wait} seconds")

    def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log training progress to file."""
        out_dir = Path(output_dir) if output_dir else Path("./.tmp/nat/finetuning/trainer_adapter")
        out_dir.mkdir(parents=True, exist_ok=True)

        log_file = out_dir / f"nemo_customizer_{ref.run_id}.jsonl"

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": ref.run_id,
            "backend": ref.backend,
            "config": {
                "namespace": self.adapter_config.namespace,
                "customization_config": self.adapter_config.customization_config,
            },
            **metrics,
        }

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        logger.debug(f"Logged progress for job {ref.run_id}")
