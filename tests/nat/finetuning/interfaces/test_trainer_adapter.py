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

import asyncio
from typing import Any

import pytest

from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.finetuning.interfaces.trainer_adapter import TrainerAdapter


class ConcreteTrainerAdapter(TrainerAdapter):
    """Concrete implementation of TrainerAdapter for testing."""

    def __init__(self, adapter_config: TrainerAdapterConfig, run_config: FinetuneRunConfig, backend: str):
        super().__init__(adapter_config, run_config, backend)
        self.initialized = False
        self.healthy = True
        self.submitted_jobs = []
        self.job_statuses = {}
        self.logged_progress = []

    async def initialize(self) -> None:
        """Initialize the adapter."""
        self.initialized = True

    async def is_healthy(self) -> bool:
        """Check if the backend is healthy."""
        return self.healthy

    async def submit(self, trajectories: TrajectoryCollection) -> TrainingJobRef:
        """Submit trajectories to the training backend."""
        job_id = f"job_{len(self.submitted_jobs)}"
        job_ref = TrainingJobRef(run_id=trajectories.run_id,
                                 backend=self.backend,
                                 metadata={
                                     "job_id": job_id, "num_trajectories": len(trajectories.trajectories)
                                 })
        self.submitted_jobs.append((trajectories, job_ref))
        self.job_statuses[job_id] = TrainingStatusEnum.RUNNING
        return job_ref

    async def status(self, ref: TrainingJobRef) -> TrainingJobStatus:
        """Get the status of a training job."""
        job_id = ref.metadata.get("job_id", "unknown")
        status = self.job_statuses.get(job_id, TrainingStatusEnum.PENDING)
        return TrainingJobStatus(run_id=ref.run_id,
                                 backend=ref.backend,
                                 status=status,
                                 progress=100.0 if status == TrainingStatusEnum.COMPLETED else 50.0,
                                 message=f"Job {job_id} status")

    async def wait_until_complete(self, ref: TrainingJobRef, poll_interval: float = 10.0) -> TrainingJobStatus:
        """Wait until the training job is complete."""
        job_id = ref.metadata.get("job_id", "unknown")
        # Simulate completion after a short delay
        await asyncio.sleep(0.1)
        self.job_statuses[job_id] = TrainingStatusEnum.COMPLETED
        return await self.status(ref)

    def log_progress(self, ref: TrainingJobRef, metrics: dict[str, Any], output_dir: str | None = None) -> None:
        """Log training adapter progress."""
        self.logged_progress.append({"ref": ref, "metrics": metrics, "output_dir": output_dir})


class TestTrainerAdapter:
    """Tests for the TrainerAdapter interface."""

    @pytest.fixture
    def adapter_config(self):
        """Create a test adapter config."""
        return TrainerAdapterConfig(type="test_adapter")

    @pytest.fixture
    def run_config(self, tmp_path):
        """Create a test run config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        return FinetuneRunConfig(config_file=config_file,
                                 target_functions=["test_function"],
                                 dataset=tmp_path / "dataset.jsonl",
                                 result_json_path="$.result")

    @pytest.fixture
    def adapter(self, adapter_config, run_config):
        """Create a concrete adapter instance."""
        return ConcreteTrainerAdapter(adapter_config=adapter_config, run_config=run_config, backend="test_backend")

    @pytest.fixture
    def sample_trajectories(self):
        """Create sample trajectories for testing."""
        return TrajectoryCollection(
            trajectories=[[Trajectory(episode=[], reward=0.8, shaped_rewards=None, metadata={"example_id": "1"})],
                          [Trajectory(episode=[], reward=0.6, shaped_rewards=None, metadata={"example_id": "2"})]],
            run_id="test_run")

    async def test_adapter_initialization(self, adapter, adapter_config, run_config):
        """Test that adapter initializes with correct configuration."""
        assert adapter.adapter_config == adapter_config
        assert adapter.run_config == run_config
        assert adapter.backend == "test_backend"

    async def test_adapter_backend_property(self, adapter):
        """Test backend property returns correct value."""
        assert adapter.backend == "test_backend"

    async def test_adapter_initialize(self, adapter):
        """Test adapter initialization."""
        assert not adapter.initialized
        await adapter.initialize()
        assert adapter.initialized

    async def test_adapter_is_healthy(self, adapter):
        """Test health check."""
        assert await adapter.is_healthy()

        adapter.healthy = False
        assert not await adapter.is_healthy()

    async def test_adapter_submit(self, adapter, sample_trajectories):
        """Test submitting trajectories."""
        job_ref = await adapter.submit(sample_trajectories)

        assert isinstance(job_ref, TrainingJobRef)
        assert job_ref.run_id == "test_run"
        assert job_ref.backend == "test_backend"
        assert "job_id" in job_ref.metadata
        assert len(adapter.submitted_jobs) == 1

    async def test_adapter_submit_multiple_jobs(self, adapter, sample_trajectories):
        """Test submitting multiple trajectory batches."""
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
        assert status.backend == "test_backend"
        assert status.status == TrainingStatusEnum.RUNNING

    async def test_adapter_wait_until_complete(self, adapter, sample_trajectories):
        """Test waiting for job completion."""
        job_ref = await adapter.submit(sample_trajectories)

        # Job should be running initially
        initial_status = await adapter.status(job_ref)
        assert initial_status.status == TrainingStatusEnum.RUNNING

        # Wait for completion
        final_status = await adapter.wait_until_complete(job_ref, poll_interval=0.01)

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
    def run_config(self, tmp_path):
        """Create a test run config."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        return FinetuneRunConfig(config_file=config_file,
                                 target_functions=["test_function"],
                                 dataset=tmp_path / "dataset.jsonl",
                                 result_json_path="$.result")

    class FailingTrainerAdapter(TrainerAdapter):
        """Adapter that fails during operations."""

        async def initialize(self) -> None:
            raise RuntimeError("Initialization failed")

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

    async def test_adapter_initialization_failure(self, failing_adapter_config, run_config):
        """Test handling of initialization failures."""
        adapter = self.FailingTrainerAdapter(failing_adapter_config, run_config, "test_backend")

        with pytest.raises(RuntimeError, match="Initialization failed"):
            await adapter.initialize()

    async def test_adapter_unhealthy_backend(self, failing_adapter_config, run_config):
        """Test handling of unhealthy backend."""
        adapter = self.FailingTrainerAdapter(failing_adapter_config, run_config, "test_backend")

        assert not await adapter.is_healthy()

    async def test_adapter_submission_failure(self, failing_adapter_config, run_config):
        """Test handling of submission failures."""
        adapter = self.FailingTrainerAdapter(failing_adapter_config, run_config, "test_backend")
        trajectories = TrajectoryCollection(trajectories=[], run_id="test_run")

        with pytest.raises(RuntimeError, match="Submission failed"):
            await adapter.submit(trajectories)

    async def test_trainer_adapter_config_reward_field(self):
        """Test that TrainerAdapterConfig has reward field that can be set."""
        from nat.data_models.finetuning import RewardFunctionConfig

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
