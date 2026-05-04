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

from typing import Any

import pytest

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import TrajectoryCollection
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter


class ConcreteTrainerAdapter(TrainerAdapter):
    """Concrete implementation of TrainerAdapter for testing."""

    def __init__(self, adapter_config: TrainerAdapterConfig):
        super().__init__(adapter_config)
        self.healthy = True
        self.submitted_jobs = []
        self.job_statuses = {}
        self.logged_progress = []

    async def is_healthy(self) -> bool:
        """Check health of backend."""
        return self.healthy

    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """Submit trajectories to remote training backend."""
        job_id = f"job_{len(self.submitted_jobs)}"
        job_ref = TrainingJobRef(run_id=trajectories.run_id,
                                 backend="test_backend",
                                 metadata={
                                     "job_id": job_id, "num_trajectories": len(trajectories.trajectories)
                                 })
        self.submitted_jobs.append((trajectories, job_ref))
        self.job_statuses[job_id] = TrainingStatusEnum.RUNNING
        return job_ref

    async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
        """Get status of a training job."""
        job_id = ref.metadata.get("job_id") if ref.metadata else None
        status = self.job_statuses.get(job_id, TrainingStatusEnum.PENDING) if job_id else TrainingStatusEnum.PENDING

        return TrainingJobStatus(run_id=ref.run_id,
                                 backend=ref.backend,
                                 status=status,
                                 progress=50.0 if status == TrainingStatusEnum.RUNNING else 100.0)

    async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float = 10.0) -> TrainingJobStatus:
        """Wait until training job completes."""
        # Simulate completion for testing
        job_id = ref.metadata.get("job_id") if ref.metadata else None
        if job_id:
            self.job_statuses[job_id] = TrainingStatusEnum.COMPLETED

        return TrainingJobStatus(run_id=ref.run_id,
                                 backend=ref.backend,
                                 status=TrainingStatusEnum.COMPLETED,
                                 progress=100.0)

    def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log training adapter progress."""
        self.logged_progress.append({"ref": ref, "metrics": metrics, "output_dir": output_dir})


class TestTrainerAdapter:
    """Tests for the TrainerAdapter interface."""

    @pytest.fixture
    def adapter_config(self):
        """Create a test adapter config."""
        return TrainerAdapterConfig(type="test_adapter", reward=RewardFunctionConfig(name="test_reward"))

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create a test finetune config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{\"input\": \"test\"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        return FinetuneConfig(run_configuration=run_config, curriculum_learning=CurriculumLearningConfig())

    @pytest.fixture
    def adapter(self, adapter_config):
        """Create a concrete adapter instance."""
        return ConcreteTrainerAdapter(adapter_config=adapter_config)

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        return TrajectoryCollection(
            trajectories=[
                [],  # Two empty trajectory groups for testing
                []
            ],
            run_id="test_run")

    async def test_adapter_initialization(self, adapter, adapter_config):
        """Test that adapter initializes with correct configuration."""
        assert adapter.adapter_config == adapter_config
        assert adapter.run_config is None

    async def test_adapter_initialize(self, adapter, finetune_config):
        """Test adapter initialization."""
        await adapter.initialize(finetune_config)
        assert adapter.run_config == finetune_config

    async def test_adapter_is_healthy(self, adapter):
        """Test health check."""
        assert await adapter.is_healthy() is True

        adapter.healthy = False
        assert await adapter.is_healthy() is False

    async def test_adapter_submit(self, adapter, sample_trajectories):
        """Test submitting trajectories."""
        job_ref = await adapter.submit(sample_trajectories)

        assert isinstance(job_ref, TrainingJobRef)
        assert job_ref.run_id == "test_run"
        assert job_ref.backend == "test_backend"
        assert len(adapter.submitted_jobs) == 1

    async def test_adapter_submit_multiple_jobs(self, adapter, sample_trajectories):
        """Test submitting multiple jobs."""
        job_ref1 = await adapter.submit(sample_trajectories)
        job_ref2 = await adapter.submit(sample_trajectories)

        assert len(adapter.submitted_jobs) == 2
        assert job_ref1.metadata["job_id"] != job_ref2.metadata["job_id"]

    async def test_adapter_status(self, adapter, sample_trajectories):
        """Test getting job status."""
        job_ref = await adapter.submit(sample_trajectories)
        status = await adapter.status(job_ref)

        assert isinstance(status, TrainingJobStatus)
        assert status.run_id == "test_run"
        assert status.status == TrainingStatusEnum.RUNNING

    async def test_adapter_wait_until_complete(self, adapter, sample_trajectories):
        """Test waiting until job completes."""
        job_ref = await adapter.submit(sample_trajectories)
        final_status = await adapter.wait_until_complete(job_ref)

        assert final_status.status == TrainingStatusEnum.COMPLETED
        assert final_status.progress == 100.0

    async def test_adapter_log_progress(self, adapter, sample_trajectories):
        """Test logging progress."""
        job_ref = await adapter.submit(sample_trajectories)
        metrics = {"loss": 0.5, "accuracy": 0.95}

        adapter.log_progress(job_ref, metrics, output_dir="/tmp/logs")

        assert len(adapter.logged_progress) == 1
        assert adapter.logged_progress[0]["ref"] == job_ref
        assert adapter.logged_progress[0]["metrics"] == metrics
        assert adapter.logged_progress[0]["output_dir"] == "/tmp/logs"

    async def test_adapter_job_metadata(self, adapter, sample_trajectories):
        """Test that job metadata is properly stored."""
        job_ref = await adapter.submit(sample_trajectories)

        assert "num_trajectories" in job_ref.metadata
        assert job_ref.metadata["num_trajectories"] == 2

    async def test_adapter_status_with_unknown_job(self, adapter):
        """Test getting status for an unknown job."""
        unknown_ref = TrainingJobRef(run_id="unknown_run", backend="test_backend", metadata={"job_id": "unknown_job"})

        status = await adapter.status(unknown_ref)
        assert status.status == TrainingStatusEnum.PENDING


class TestTrainerAdapterErrorHandling:
    """Tests for TrainerAdapter error handling and edge cases."""

    @pytest.fixture
    def failing_adapter_config(self):
        """Create an adapter config that might fail."""
        return TrainerAdapterConfig(type="failing_adapter")

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create a test finetune config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{\"input\": \"test\"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        return FinetuneConfig(run_configuration=run_config)

    class FailingTrainerAdapter(TrainerAdapter):
        """Adapter that fails during operations."""

        async def is_healthy(self) -> bool:
            return False

        async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
            raise RuntimeError("Submission failed")

        async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
            raise RuntimeError("Status check failed")

        async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float = 10.0) -> TrainingJobStatus:
            raise RuntimeError("Wait failed")

        def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
            raise RuntimeError("Logging failed")

    async def test_adapter_unhealthy_backend(self, failing_adapter_config):
        """Test handling of unhealthy backend."""
        adapter = self.FailingTrainerAdapter(failing_adapter_config)

        assert not await adapter.is_healthy()

    async def test_adapter_submission_failure(self, failing_adapter_config):
        """Test handling of submission failures."""
        adapter = self.FailingTrainerAdapter(failing_adapter_config)
        trajectories = TrajectoryCollection(trajectories=[], run_id="test_run")

        with pytest.raises(RuntimeError, match="Submission failed"):
            await adapter.submit(trajectories)

    async def test_trainer_adapter_config_reward_field(self):
        """Test that TrainerAdapterConfig has reward field that can be set."""

        class TestTrainerAdapterConfig(TrainerAdapterConfig, name="test_adapter_with_reward"):
            pass

        config = TestTrainerAdapterConfig(reward=RewardFunctionConfig(name="test_reward"))
        assert config.reward is not None
        assert isinstance(config.reward, RewardFunctionConfig)
        assert config.reward.name == "test_reward"

    async def test_trainer_adapter_config_reward_field_default(self):
        """Test that TrainerAdapterConfig reward field defaults to None."""

        class TestTrainerAdapterConfig(TrainerAdapterConfig, name="test_adapter_no_reward"):
            pass

        config = TestTrainerAdapterConfig()
        assert config.reward is None
