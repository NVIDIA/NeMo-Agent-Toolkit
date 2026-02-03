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

import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import CurriculumLearningConfig
from nat.data_models.finetuning import EpisodeItem
from nat.data_models.finetuning import EpisodeItemRole
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.plugins.openpipe.config import ARTBackendConfig
from nat.plugins.openpipe.config import ARTTrainerAdapterConfig
from nat.plugins.openpipe.trainer_adapter import ARTTrainerAdapter


class TestARTTrainerAdapter:
    """Comprehensive tests for ARTTrainerAdapter implementation."""

    @pytest.fixture
    def backend_config(self):
        """Create test backend configuration."""
        return ARTBackendConfig(ip="127.0.0.1",
                                port=8000,
                                name="test_trainer",
                                project="test_project",
                                base_model="test_model",
                                api_key="test_key",
                                delete_old_checkpoints=False,
                                init_args=None,
                                engine_args=None,
                                torchtune_args=None,
                                server_config=None)

    @pytest.fixture
    def adapter_config(self, backend_config):
        """Create test adapter configuration."""
        return ARTTrainerAdapterConfig(backend=backend_config,
                                       training=None,
                                       reward=RewardFunctionConfig(name="test_reward"))

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create test finetune configuration."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{"input": "test"}')

        run_config = FinetuneRunConfig(config_file=config_file,
                                       target_functions=["test_function"],
                                       dataset=str(dataset_file),
                                       result_json_path="$.result")

        return FinetuneConfig(run_configuration=run_config,
                              curriculum_learning=CurriculumLearningConfig(),
                              reward_function=RewardFunctionConfig(name="test_reward"))

    @pytest.fixture
    def mock_art_backend(self):
        """Create mock ART Backend."""
        with patch('nat.plugins.openpipe.trainer_adapter.art.Backend') as mock:
            yield mock.return_value

    @pytest.fixture
    def mock_art_model(self):
        """Create mock ART TrainableModel."""
        with patch('nat.plugins.openpipe.trainer_adapter.art.TrainableModel') as mock:
            model_instance = mock.return_value
            model_instance.register = AsyncMock()
            model_instance.train = AsyncMock()
            model_instance.delete_checkpoints = AsyncMock()
            yield model_instance

    @pytest.fixture
    def adapter(self, adapter_config, mock_art_backend, mock_art_model):
        """Create ARTTrainerAdapter instance with mocked dependencies."""
        with patch('nat.plugins.openpipe.trainer_adapter.art.dev.InternalModelConfig'):
            adapter = ARTTrainerAdapter(adapter_config=adapter_config)
            adapter.model = mock_art_model
            adapter.remote_backend = mock_art_backend
            return adapter

    def test_adapter_initialization(self, adapter, adapter_config):
        """Test adapter initializes with correct configuration."""
        assert adapter.adapter_config == adapter_config
        assert adapter._training_jobs == {}
        assert adapter.model is not None
        assert adapter.remote_backend is not None

    async def test_adapter_initialize(self, adapter, finetune_config):
        """Test adapter initialization with finetune config."""
        with patch.object(adapter, 'is_healthy', return_value=True):
            await adapter.initialize(finetune_config)

        assert adapter.run_config == finetune_config
        assert adapter.adapter_config.reward == finetune_config.reward_function
        adapter.model.register.assert_called_once()

    async def test_adapter_initialize_unhealthy_backend(self, adapter, finetune_config):
        """Test adapter initialization with unhealthy backend."""
        with patch.object(adapter, 'is_healthy', return_value=False):
            with pytest.raises(ConnectionError, match="Failed to connect to ART backend"):
                await adapter.initialize(finetune_config)

    @patch('httpx.AsyncClient')
    async def test_is_healthy_success(self, mock_client, adapter):
        """Test health check success."""
        mock_response = MagicMock()
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        result = await adapter.is_healthy()
        assert result is True

    @patch('httpx.AsyncClient')
    async def test_is_healthy_failure(self, mock_client, adapter):
        """Test health check failure."""
        import httpx
        mock_client.return_value.__aenter__.return_value.get = AsyncMock(
            side_effect=httpx.HTTPError("Connection failed"))

        result = await adapter.is_healthy()
        assert result is False

    async def test_validate_episode_order_valid(self, adapter):
        """Test validation of valid trajectory episode order."""
        trajectory = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Hello"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Hi", logprobs={"test": 0.5})
        ],
                                reward=0.8)

        # Should not raise exception
        await adapter._validate_episode_order(trajectory)

    async def test_validate_episode_order_empty(self, adapter):
        """Test validation of empty trajectory."""
        trajectory = Trajectory(episode=[], reward=0.8)

        with pytest.raises(ValueError, match="Trajectory episode is empty"):
            await adapter._validate_episode_order(trajectory)

    async def test_validate_episode_order_invalid_first(self, adapter):
        """Test validation with invalid first message."""
        trajectory = Trajectory(
            episode=[EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Hi", logprobs={"test": 0.5})], reward=0.8)

        with pytest.raises(ValueError, match="first message.*must be from 'user' or 'system'"):
            await adapter._validate_episode_order(trajectory)

    async def test_validate_episode_order_consecutive_assistant(self, adapter):
        """Test validation with consecutive assistant messages."""
        trajectory = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Hello"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Hi", logprobs={"test": 0.5}),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="How are you?", logprobs={"test": 0.6})
        ],
                                reward=0.8)

        with pytest.raises(ValueError, match="Consecutive assistant messages"):
            await adapter._validate_episode_order(trajectory)

    @patch('nat.plugins.openpipe.trainer_adapter.art.Trajectory')
    @patch('nat.plugins.openpipe.trainer_adapter.art.TrajectoryGroup')
    async def test_construct_trajectory_groups(self, mock_traj_group, mock_art_traj, adapter):
        """Test construction of ART trajectory groups from NAT trajectories."""
        # Create NAT trajectories
        trajectory1 = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Question 1"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Answer 1", logprobs={"test": 0.5})
        ],
                                 reward=0.8)
        trajectory2 = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.SYSTEM, content="System prompt"),
            EpisodeItem(role=EpisodeItemRole.USER, content="Question 2"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Answer 2", logprobs={"test": 0.6})
        ],
                                 reward=0.9)

        trajectory_lists = [[trajectory1], [trajectory2]]

        # Mock ART trajectory creation
        mock_art_traj.return_value.model_validate = MagicMock()

        # Construct groups
        result = await adapter._construct_trajectory_groups(trajectory_lists)

        # Verify trajectory groups were created
        assert len(result) == 2
        assert mock_traj_group.call_count == 2

    @patch('nat.plugins.openpipe.trainer_adapter.art.Trajectory')
    async def test_construct_trajectory_groups_with_invalid(self, mock_art_traj, adapter):
        """Test construction skips invalid trajectories."""
        # Create invalid trajectory (will cause exception during construction)
        trajectory = Trajectory(
            episode=[
                EpisodeItem(role=EpisodeItemRole.USER, content="Question")
                # Missing assistant message
            ],
            reward=0.8)

        trajectory_lists = [[trajectory]]

        # Mock ART trajectory to raise exception
        mock_art_traj.side_effect = Exception("Invalid trajectory")

        # Construct groups - should return empty list
        result = await adapter._construct_trajectory_groups(trajectory_lists)
        assert len(result) == 0

    async def test_submit_trajectories(self, adapter):
        """Test submitting trajectories for training."""
        # Create trajectories
        trajectory = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Question"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Answer", logprobs={"test": 0.5})
        ],
                                reward=0.8)
        collection = TrajectoryCollection(trajectories=[[trajectory]], run_id="test_run")

        # Mock trajectory group construction
        mock_group = MagicMock()
        mock_group.trajectories = [MagicMock()]
        with patch.object(adapter, '_construct_trajectory_groups', return_value=[mock_group]):
            # Submit trajectories
            job_ref = await adapter.submit(collection)

        assert job_ref.run_id == "test_run"
        assert job_ref.backend == "openpipe-art"
        assert "test_run" in adapter._training_jobs
        adapter.model.train.assert_called_once()

    async def test_submit_no_valid_trajectories(self, adapter):
        """Test submitting with no valid trajectories."""
        collection = TrajectoryCollection(trajectories=[], run_id="test_run")

        with patch.object(adapter, '_construct_trajectory_groups', return_value=[]):
            with pytest.raises(ValueError, match="No valid trajectory groups"):
                await adapter.submit(collection)

    async def test_submit_duplicate_run_id(self, adapter):
        """Test submitting with duplicate run ID."""
        # Create a valid trajectory
        trajectory = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Question"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Answer", logprobs={"test": 0.5})
        ],
                                reward=0.8)
        collection = TrajectoryCollection(trajectories=[[trajectory]], run_id="test_run")

        # Mock successful trajectory group construction
        mock_group = MagicMock()
        mock_group.trajectories = [MagicMock()]
        with patch.object(adapter, '_construct_trajectory_groups', return_value=[mock_group]):
            adapter._training_jobs["test_run"] = MagicMock()

            with pytest.raises(AssertionError, match="Training job.*already exists"):
                await adapter.submit(collection)

    async def test_submit_with_checkpoint_deletion(self, adapter):
        """Test submitting with old checkpoint deletion."""
        adapter.adapter_config.backend.delete_old_checkpoints = True

        trajectory = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Question"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Answer", logprobs={"test": 0.5})
        ],
                                reward=0.8)
        collection = TrajectoryCollection(trajectories=[[trajectory]], run_id="test_run")

        mock_group = MagicMock()
        mock_group.trajectories = [MagicMock()]
        with patch.object(adapter, '_construct_trajectory_groups', return_value=[mock_group]):
            await adapter.submit(collection)

        adapter.model.delete_checkpoints.assert_called_once()

    async def test_status_running_job(self, adapter):
        """Test getting status of running job."""
        # Create mock task
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = False
        adapter._training_jobs["test_run"] = task

        ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        status = await adapter.status(ref)

        assert status.status == TrainingStatusEnum.RUNNING
        assert status.message == "Training is in progress."
        assert "test_run" in adapter._training_jobs  # Should not be removed

    async def test_status_completed_job(self, adapter):
        """Test getting status of completed job."""
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        task.cancelled.return_value = False
        task.exception.return_value = None
        adapter._training_jobs["test_run"] = task

        ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        status = await adapter.status(ref)

        assert status.status == TrainingStatusEnum.COMPLETED
        assert status.progress == 100.0
        assert "test_run" not in adapter._training_jobs  # Should be removed

    async def test_status_failed_job(self, adapter):
        """Test getting status of failed job."""
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        task.cancelled.return_value = False
        task.exception.return_value = Exception("Training failed")
        adapter._training_jobs["test_run"] = task

        ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        status = await adapter.status(ref)

        assert status.status == TrainingStatusEnum.FAILED
        assert "Training failed" in status.message
        assert "test_run" not in adapter._training_jobs  # Should be removed

    async def test_status_cancelled_job(self, adapter):
        """Test getting status of cancelled job."""
        task = MagicMock(spec=asyncio.Task)
        task.done.return_value = True
        task.cancelled.return_value = True
        adapter._training_jobs["test_run"] = task

        ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        status = await adapter.status(ref)

        assert status.status == TrainingStatusEnum.CANCELED
        assert status.message == "Training was cancelled."
        assert "test_run" not in adapter._training_jobs  # Should be removed

    async def test_status_unknown_job(self, adapter):
        """Test getting status of unknown job."""
        ref = TrainingJobRef(run_id="unknown_run", backend="openpipe-art")

        with pytest.raises(ValueError, match="No training job found"):
            await adapter.status(ref)

    async def test_wait_until_complete(self, adapter):
        """Test waiting for job completion."""
        # Create mock task that completes after one check
        task = MagicMock(spec=asyncio.Task)
        # First call to done() returns False, second returns True
        done_call_count = [0]

        def done_side_effect():
            done_call_count[0] += 1
            return done_call_count[0] > 1

        task.done = done_side_effect
        task.cancelled = MagicMock(return_value=False)
        task.exception = MagicMock(return_value=None)
        adapter._training_jobs["test_run"] = task

        ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")

        with patch('asyncio.sleep', new_callable=AsyncMock):
            status = await adapter.wait_until_complete(ref, poll_interval=0.01)

        assert status.status == TrainingStatusEnum.COMPLETED

    async def test_wait_until_complete_unknown_job(self, adapter):
        """Test waiting for unknown job."""
        ref = TrainingJobRef(run_id="unknown_run", backend="openpipe-art")

        with pytest.raises(ValueError, match="No training job found"):
            await adapter.wait_until_complete(ref)

    def test_log_progress(self, adapter, tmp_path):
        """Test logging training progress."""
        ref = TrainingJobRef(run_id="test_run", backend="openpipe-art")
        metrics = {"status": TrainingStatusEnum.RUNNING, "progress": 50.0, "current_loss": 0.5}

        output_dir = tmp_path / "logs"
        adapter.log_progress(ref, metrics, output_dir=str(output_dir))

        # Check log file was created
        log_file = output_dir / f"trainer_adapter_{ref.run_id}.jsonl"
        assert log_file.exists()

        # Verify log content
        with open(log_file) as f:
            log_entry = json.loads(f.readline())
            assert log_entry["run_id"] == "test_run"
            assert log_entry["backend"] == "openpipe-art"
            assert log_entry["status"] == TrainingStatusEnum.RUNNING
            assert log_entry["progress"] == 50.0

    def test_training_jobs_property(self, adapter):
        """Test training_jobs property."""
        task = MagicMock()
        adapter._training_jobs["test_run"] = task

        jobs = adapter.training_jobs
        assert "test_run" in jobs
        assert jobs["test_run"] == task

    @patch('nat.plugins.openpipe.trainer_adapter.art.Trajectory')
    @patch('nat.plugins.openpipe.trainer_adapter.art.TrajectoryGroup')
    async def test_construct_trajectory_with_tool_calls(self, mock_traj_group, mock_art_traj, adapter):
        """Test constructing trajectories with tool/function calls."""
        trajectory = Trajectory(episode=[
            EpisodeItem(role=EpisodeItemRole.USER, content="Use a tool"),
            EpisodeItem(role=EpisodeItemRole.TOOL, content="Tool result"),
            EpisodeItem(role=EpisodeItemRole.ASSISTANT, content="Based on tool", logprobs={"test": 0.5})
        ],
                                reward=0.7)

        trajectory_lists = [[trajectory]]
        mock_art_traj.return_value.model_validate = MagicMock()

        result = await adapter._construct_trajectory_groups(trajectory_lists)

        assert len(result) == 1
        mock_art_traj.assert_called()

    async def test_submit_task_callback_success(self, adapter):
        """Test task callback on successful training."""
        collection = TrajectoryCollection(trajectories=[[]], run_id="test_run")

        # Mock trajectory group construction
        mock_group = MagicMock()
        mock_group.trajectories = [MagicMock()]

        with patch.object(adapter, '_construct_trajectory_groups', return_value=[mock_group]):
            # Create custom mock task to control callback
            task = MagicMock(spec=asyncio.Task)
            task.cancelled.return_value = False
            task.exception.return_value = None
            callbacks = []

            def add_callback(cb):
                callbacks.append(cb)
                # Simulate immediate completion
                cb(task)

            task.add_done_callback = add_callback

            with patch('asyncio.create_task', return_value=task):
                await adapter.submit(collection)

        # Verify callback was added
        assert len(callbacks) == 1

    async def test_submit_task_callback_failure(self, adapter):
        """Test task callback on failed training."""
        collection = TrajectoryCollection(trajectories=[[]], run_id="test_run")

        # Mock trajectory group construction
        mock_group = MagicMock()
        mock_group.trajectories = [MagicMock()]

        with patch.object(adapter, '_construct_trajectory_groups', return_value=[mock_group]):
            # Create custom mock task to control callback
            task = MagicMock(spec=asyncio.Task)
            task.cancelled.return_value = False
            task.exception.return_value = Exception("Training error")
            callbacks = []

            def add_callback(cb):
                callbacks.append(cb)
                # Simulate failure
                cb(task)

            task.add_done_callback = add_callback

            with patch('asyncio.create_task', return_value=task):
                await adapter.submit(collection)

        # Verify callback was added and handled exception
        assert len(callbacks) == 1
