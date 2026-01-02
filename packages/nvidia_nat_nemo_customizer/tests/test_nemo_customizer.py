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
"""Tests for NeMo Customizer TrainerAdapter and Trainer."""

import json
import uuid
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import DPOItem
from nat.data_models.finetuning import FinetuneConfig
from nat.data_models.finetuning import FinetuneRunConfig
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.finetuning import RewardFunctionConfig
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingJobStatus
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.plugins.customizer.dpo.config import DPOSpecificHyperparameters
from nat.plugins.customizer.dpo.config import NeMoCustomizerHyperparameters
from nat.plugins.customizer.dpo.config import NeMoCustomizerTrainerAdapterConfig
from nat.plugins.customizer.dpo.config import NeMoCustomizerTrainerConfig
from nat.plugins.customizer.dpo.config import NIMDeploymentConfig
from nat.plugins.customizer.dpo.trainer import NeMoCustomizerTrainer
from nat.plugins.customizer.dpo.trainer_adapter import NeMoCustomizerTrainerAdapter

# =============================================================================
# Configuration Tests
# =============================================================================


class TestNeMoCustomizerHyperparameters:
    """Tests for hyperparameter configuration."""

    def test_default_values(self):
        """Test default hyperparameter values."""
        hp = NeMoCustomizerHyperparameters()

        assert hp.training_type == "dpo"
        assert hp.finetuning_type == "all_weights"
        assert hp.epochs == 3
        assert hp.batch_size == 4
        assert hp.learning_rate == 5e-5
        assert hp.dpo.ref_policy_kl_penalty == 0.1

    def test_custom_values(self):
        """Test custom hyperparameter values."""
        hp = NeMoCustomizerHyperparameters(
            training_type="sft",
            finetuning_type="lora",
            epochs=10,
            batch_size=16,
            learning_rate=1e-4,
            dpo=DPOSpecificHyperparameters(ref_policy_kl_penalty=0.2),
        )

        assert hp.training_type == "sft"
        assert hp.finetuning_type == "lora"
        assert hp.epochs == 10
        assert hp.batch_size == 16
        assert hp.learning_rate == 1e-4
        assert hp.dpo.ref_policy_kl_penalty == 0.2

    def test_invalid_epochs(self):
        """Test invalid epochs raises error."""
        with pytest.raises(ValueError):
            NeMoCustomizerHyperparameters(epochs=0)

    def test_invalid_learning_rate(self):
        """Test invalid learning rate raises error."""
        with pytest.raises(ValueError):
            NeMoCustomizerHyperparameters(learning_rate=0.0)


class TestNIMDeploymentConfig:
    """Tests for NIM deployment configuration."""

    def test_default_values(self):
        """Test default deployment config values."""
        config = NIMDeploymentConfig()

        assert config.image_name == "nvcr.io/nim/meta/llama-3.1-8b-instruct"
        assert config.image_tag == "latest"
        assert config.gpu == 1
        assert config.deployment_name is None
        assert config.description == "Fine-tuned model deployment"

    def test_custom_values(self):
        """Test custom deployment config values."""
        config = NIMDeploymentConfig(
            image_name="nvcr.io/nim/meta/llama-3.1-8b-instruct",
            image_tag="v1.0.0",
            gpu=4,
            deployment_name="my-deployment",
            description="Custom deployment",
        )

        assert config.image_name == "nvcr.io/nim/meta/llama-3.1-8b-instruct"
        assert config.image_tag == "v1.0.0"
        assert config.gpu == 4
        assert config.deployment_name == "my-deployment"
        assert config.description == "Custom deployment"


class TestNeMoCustomizerTrainerAdapterConfig:
    """Tests for TrainerAdapter configuration."""

    def test_required_fields(self):
        """Test required fields are validated."""
        with pytest.raises(ValueError):
            NeMoCustomizerTrainerAdapterConfig()

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config = NeMoCustomizerTrainerAdapterConfig(
            entity_host="https://nmp.example.com",
            datastore_host="https://datastore.example.com",
            namespace="test-namespace",
            customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
        )

        assert config.entity_host == "https://nmp.example.com"
        assert config.datastore_host == "https://datastore.example.com"
        assert config.namespace == "test-namespace"
        assert config.customization_config == "meta/llama-3.2-1b-instruct@v1.0.0+A100"
        assert config.use_full_message_history is False
        assert config.deploy_on_completion is False

    def test_trailing_slash_removed(self):
        """Test trailing slashes are removed from hosts."""
        config = NeMoCustomizerTrainerAdapterConfig(
            entity_host="https://nmp.example.com/",
            datastore_host="https://datastore.example.com/",
            namespace="test-namespace",
            customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
        )

        assert config.entity_host == "https://nmp.example.com"
        assert config.datastore_host == "https://datastore.example.com"

    def test_full_config(self):
        """Test full configuration with all options."""
        config = NeMoCustomizerTrainerAdapterConfig(
            entity_host="https://nmp.example.com",
            datastore_host="https://datastore.example.com",
            hf_token="my-token",
            namespace="test-namespace",
            dataset_name="my-dataset",
            dataset_output_dir="/path/to/datasets",
            create_namespace_if_missing=False,
            customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
            hyperparameters=NeMoCustomizerHyperparameters(epochs=5),
            use_full_message_history=False,
            deploy_on_completion=True,
            deployment_config=NIMDeploymentConfig(gpu=2),
            poll_interval_seconds=60.0,
        )

        assert config.hf_token == "my-token"
        assert config.dataset_name == "my-dataset"
        assert config.dataset_output_dir == "/path/to/datasets"
        assert config.create_namespace_if_missing is False
        assert config.hyperparameters.epochs == 5
        assert config.use_full_message_history is False
        assert config.deploy_on_completion is True
        assert config.deployment_config.gpu == 2
        assert config.poll_interval_seconds == 60.0

    def test_dataset_output_dir_default_none(self):
        """Test dataset_output_dir defaults to None."""
        config = NeMoCustomizerTrainerAdapterConfig(
            entity_host="https://nmp.example.com",
            datastore_host="https://datastore.example.com",
            namespace="test-namespace",
            customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
        )

        assert config.dataset_output_dir is None

    def test_config_name(self):
        """Test config is registered with correct name."""
        assert NeMoCustomizerTrainerAdapterConfig._typed_model_name == "nemo_customizer_trainer_adapter"


# =============================================================================
# TrainerAdapter Tests
# =============================================================================


@pytest.fixture
def adapter_config():
    """Create a test adapter configuration."""
    return NeMoCustomizerTrainerAdapterConfig(
        entity_host="https://nmp.example.com",
        datastore_host="https://datastore.example.com",
        namespace="test-namespace",
        customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
    )


@pytest.fixture
def trainer_adapter(adapter_config):
    """Create a trainer adapter instance."""
    return NeMoCustomizerTrainerAdapter(adapter_config=adapter_config)


@pytest.fixture
def sample_trajectories():
    """Create sample trajectory collection with DPO items."""
    dpo_item1 = DPOItem(
        prompt=[
            OpenAIMessage(role="system", content="You are a helpful assistant."),
            OpenAIMessage(role="user", content="What is 2+2?"),
        ],
        chosen_response="The answer is 4.",
        rejected_response="I don't know.",
    )
    dpo_item2 = DPOItem(
        prompt="Simple prompt",
        chosen_response="Good response",
        rejected_response="Bad response",
    )

    trajectories = [
        [Trajectory(episode=[dpo_item1], reward=0.9, metadata={"example_id": "ex_1"})],
        [Trajectory(episode=[dpo_item2], reward=0.8, metadata={"example_id": "ex_2"})],
    ]

    return TrajectoryCollection(trajectories=trajectories, run_id="test-run-123")


class TestNeMoCustomizerTrainerAdapter:
    """Tests for NeMoCustomizerTrainerAdapter."""

    def test_initialization(self, trainer_adapter, adapter_config):
        """Test adapter initialization."""
        assert trainer_adapter.adapter_config == adapter_config
        assert trainer_adapter._entity_client is None
        assert trainer_adapter._hf_api is None
        assert len(trainer_adapter._active_jobs) == 0

    def test_lazy_client_initialization(self, trainer_adapter):
        """Test lazy initialization of clients."""
        # Clients should be None initially
        assert trainer_adapter._entity_client is None
        assert trainer_adapter._hf_api is None

        # Accessing entity_client should initialize it
        with patch("nat.plugins.customizer.dpo.trainer_adapter.NeMoMicroservices") as mock_client:
            _ = trainer_adapter.entity_client
            mock_client.assert_called_once_with(base_url="https://nmp.example.com")

    def test_format_prompt_full_history_with_messages(self, trainer_adapter):
        """Test prompt formatting with full message history."""
        trainer_adapter.adapter_config.use_full_message_history = True

        messages = [
            OpenAIMessage(role="system", content="System message"),
            OpenAIMessage(role="user", content="User message"),
        ]

        result = trainer_adapter._format_prompt(messages)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == {"role": "system", "content": "System message"}
        assert result[1] == {"role": "user", "content": "User message"}

    def test_format_prompt_full_history_with_string(self, trainer_adapter):
        """Test prompt formatting with string prompt in full history mode."""
        trainer_adapter.adapter_config.use_full_message_history = True

        result = trainer_adapter._format_prompt("Simple prompt")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Simple prompt"}

    def test_format_prompt_last_message_only_with_messages(self, trainer_adapter):
        """Test prompt formatting with last message only."""
        trainer_adapter.adapter_config.use_full_message_history = False

        messages = [
            OpenAIMessage(role="system", content="System message"),
            OpenAIMessage(role="user", content="User message"),
        ]

        result = trainer_adapter._format_prompt(messages)

        assert isinstance(result, str)
        assert result == "User message"

    def test_format_prompt_last_message_only_with_string(self, trainer_adapter):
        """Test prompt formatting with string in last message mode."""
        trainer_adapter.adapter_config.use_full_message_history = False

        result = trainer_adapter._format_prompt("Simple prompt")

        assert result == "Simple prompt"

    def test_format_prompt_empty_messages(self, trainer_adapter):
        """Test prompt formatting with empty message list."""
        trainer_adapter.adapter_config.use_full_message_history = False

        result = trainer_adapter._format_prompt([])

        assert result == ""

    def test_trajectory_to_dpo_jsonl(self, trainer_adapter, sample_trajectories):
        """Test converting trajectories to JSONL format."""
        trainer_adapter.adapter_config.use_full_message_history = True

        training_jsonl, validation_jsonl = trainer_adapter._trajectory_to_dpo_jsonl(sample_trajectories)

        # Parse and verify training data
        training_lines = training_jsonl.strip().split("\n")
        assert len(training_lines) >= 1

        first_item = json.loads(training_lines[0])
        assert "prompt" in first_item
        assert "chosen_response" in first_item
        assert "rejected_response" in first_item

        # Verify validation data exists
        validation_lines = validation_jsonl.strip().split("\n")
        assert len(validation_lines) >= 1

    def test_trajectory_to_dpo_jsonl_last_message_mode(self, trainer_adapter, sample_trajectories):
        """Test JSONL conversion with last message mode."""
        trainer_adapter.adapter_config.use_full_message_history = False

        training_jsonl, _ = trainer_adapter._trajectory_to_dpo_jsonl(sample_trajectories)

        # Parse and verify format
        training_lines = training_jsonl.strip().split("\n")
        first_item = json.loads(training_lines[0])

        # Prompt should be the last message content as string
        assert isinstance(first_item["prompt"], str)

    def test_trajectory_to_dpo_jsonl_empty_raises(self, trainer_adapter):
        """Test that empty trajectories raise error."""
        empty_collection = TrajectoryCollection(trajectories=[], run_id="empty-run")

        with pytest.raises(ValueError, match="No DPO items found"):
            trainer_adapter._trajectory_to_dpo_jsonl(empty_collection)

    async def test_is_healthy_success(self, trainer_adapter):
        """Test health check success."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_client.get.return_value = mock_response

            result = await trainer_adapter.is_healthy()

            assert result is True

    async def test_is_healthy_failure(self, trainer_adapter):
        """Test health check always returns True (stub implementation)."""
        # Note: Current implementation is a stub that always returns True
        # This test verifies the stub behavior
        result = await trainer_adapter.is_healthy()

        assert result is True

    async def test_submit_creates_job(self, trainer_adapter, sample_trajectories):
        """Test submitting trajectories creates a job."""
        with patch.object(trainer_adapter, "_setup_dataset", new_callable=AsyncMock) as mock_setup:
            mock_setup.return_value = "test-dataset-123"

            mock_job = MagicMock()
            mock_job.id = "job-123"
            mock_job.output_model = "default/model@job-123"

            mock_entity_client = MagicMock()
            mock_entity_client.customization.jobs.create.return_value = mock_job
            trainer_adapter._entity_client = mock_entity_client

            ref = await trainer_adapter.submit(sample_trajectories)

            assert ref.run_id == "test-run-123"
            assert ref.backend == "nemo-customizer"
            assert ref.metadata["job_id"] == "job-123"
            assert ref.metadata["output_model"] == "default/model@job-123"
            assert "test-run-123" in trainer_adapter._active_jobs

    async def test_submit_duplicate_run_raises(self, trainer_adapter, sample_trajectories):
        """Test submitting duplicate run raises error."""
        trainer_adapter._active_jobs["test-run-123"] = "existing-job"

        with pytest.raises(ValueError, match="already exists"):
            await trainer_adapter.submit(sample_trajectories)

    async def test_status_returns_job_status(self, trainer_adapter):
        """Test getting job status."""
        trainer_adapter._active_jobs["test-run"] = "job-123"
        trainer_adapter._job_output_models["test-run"] = "output-model"

        mock_job_status = MagicMock()
        mock_job_status.status = "running"
        mock_job_status.percentage_done = 50.0
        mock_job_status.epochs_completed = 1

        mock_entity_client = MagicMock()
        mock_entity_client.customization.jobs.status.return_value = mock_job_status
        trainer_adapter._entity_client = mock_entity_client

        ref = TrainingJobRef(run_id="test-run", backend="nemo-customizer")
        status = await trainer_adapter.status(ref)

        assert status.run_id == "test-run"
        assert status.status == TrainingStatusEnum.RUNNING
        assert status.progress == 50.0

    async def test_status_unknown_run_uses_metadata(self, trainer_adapter):
        """Test status lookup uses metadata when run not in active jobs."""
        mock_job_status = MagicMock()
        mock_job_status.status = "completed"
        mock_job_status.percentage_done = 100.0

        mock_entity_client = MagicMock()
        mock_entity_client.customization.jobs.status.return_value = mock_job_status
        trainer_adapter._entity_client = mock_entity_client

        ref = TrainingJobRef(
            run_id="unknown-run",
            backend="nemo-customizer",
            metadata={"job_id": "job-from-metadata"},
        )
        status = await trainer_adapter.status(ref)

        assert status.status == TrainingStatusEnum.COMPLETED
        mock_entity_client.customization.jobs.status.assert_called_once_with("job-from-metadata")

    async def test_status_unknown_run_no_metadata_raises(self, trainer_adapter):
        """Test status with unknown run and no metadata raises error."""
        ref = TrainingJobRef(run_id="unknown-run", backend="nemo-customizer")

        with pytest.raises(ValueError, match="No training job found"):
            await trainer_adapter.status(ref)

    def test_log_progress(self, trainer_adapter, tmp_path):
        """Test logging progress to file."""
        ref = TrainingJobRef(run_id="test-run", backend="nemo-customizer")

        trainer_adapter.log_progress(
            ref=ref,
            metrics={
                "status": "running", "progress": 50
            },
            output_dir=str(tmp_path),
        )

        log_file = tmp_path / "nemo_customizer_test-run.jsonl"
        assert log_file.exists()

        with open(log_file) as f:
            log_entry = json.loads(f.readline())

        assert log_entry["run_id"] == "test-run"
        assert log_entry["backend"] == "nemo-customizer"
        assert log_entry["status"] == "running"
        assert log_entry["progress"] == 50


class TestTrainerAdapterIntegration:
    """Integration-style tests for the trainer adapter."""

    async def test_full_workflow_mock(self, adapter_config, sample_trajectories):
        """Test full workflow with mocked external services."""
        adapter = NeMoCustomizerTrainerAdapter(adapter_config=adapter_config)

        # Mock all external dependencies
        mock_entity_client = MagicMock()
        mock_hf_api = MagicMock()

        adapter._entity_client = mock_entity_client
        adapter._hf_api = mock_hf_api

        # Mock job creation
        mock_job = MagicMock()
        mock_job.id = "cust-ABC123"
        mock_job.output_model = "default/model@cust-ABC123"
        mock_entity_client.customization.jobs.create.return_value = mock_job

        # Mock HF API calls
        mock_hf_api.create_repo.return_value = None
        mock_hf_api.upload_file.return_value = None
        mock_entity_client.datasets.create.return_value = None

        # Submit job
        ref = await adapter.submit(sample_trajectories)

        assert ref.run_id == sample_trajectories.run_id
        assert ref.backend == "nemo-customizer"
        assert "cust-ABC123" in ref.metadata["job_id"]

        # Verify dataset was created
        mock_hf_api.create_repo.assert_called_once()
        assert mock_hf_api.upload_file.call_count == 2  # train + validation

        # Verify job was created with correct params
        mock_entity_client.customization.jobs.create.assert_called_once()
        call_kwargs = mock_entity_client.customization.jobs.create.call_args[1]
        assert call_kwargs["config"] == adapter_config.customization_config
        assert call_kwargs["dataset"]["namespace"] == adapter_config.namespace

    async def test_submit_with_dataset_output_dir(self, sample_trajectories, tmp_path):
        """Test that dataset files are saved to configured output directory."""
        config = NeMoCustomizerTrainerAdapterConfig(
            entity_host="https://nmp.example.com",
            datastore_host="https://datastore.example.com",
            namespace="test-namespace",
            customization_config="meta/llama-3.2-1b-instruct@v1.0.0+A100",
            dataset_output_dir=str(tmp_path),
        )
        adapter = NeMoCustomizerTrainerAdapter(adapter_config=config)

        # Mock all external dependencies
        mock_entity_client = MagicMock()
        mock_hf_api = MagicMock()

        adapter._entity_client = mock_entity_client
        adapter._hf_api = mock_hf_api

        # Mock job creation
        mock_job = MagicMock()
        mock_job.id = "cust-ABC123"
        mock_job.output_model = "default/model@cust-ABC123"
        mock_entity_client.customization.jobs.create.return_value = mock_job

        # Mock HF API calls
        mock_hf_api.create_repo.return_value = None
        mock_hf_api.upload_file.return_value = None
        mock_entity_client.datasets.create.return_value = None

        # Submit job
        await adapter.submit(sample_trajectories)

        # Verify dataset files were saved to the configured directory
        run_dir = tmp_path / sample_trajectories.run_id
        assert run_dir.exists()

        train_file = run_dir / "training_file.jsonl"
        val_file = run_dir / "validation_file.jsonl"

        assert train_file.exists()
        assert val_file.exists()

        # Verify content is valid JSONL
        with open(train_file) as f:
            first_line = json.loads(f.readline())
            assert "prompt" in first_line
            assert "chosen_response" in first_line
            assert "rejected_response" in first_line


# =============================================================================
# Trainer Configuration Tests
# =============================================================================


class TestNeMoCustomizerTrainerConfig:
    """Tests for NeMo Customizer Trainer configuration."""

    def test_default_values(self):
        """Test default trainer config values."""
        config = NeMoCustomizerTrainerConfig(reward=RewardFunctionConfig(name="test_reward"))

        assert config.num_runs == 1
        assert config.continue_on_collection_error is False
        assert config.deduplicate_pairs is True
        assert config.max_pairs is None
        assert config.wait_for_completion is True

    def test_custom_values(self):
        """Test custom trainer config values."""
        config = NeMoCustomizerTrainerConfig(
            reward=RewardFunctionConfig(name="test_reward"),
            num_runs=5,
            continue_on_collection_error=True,
            deduplicate_pairs=False,
            max_pairs=1000,
            wait_for_completion=False,
        )

        assert config.num_runs == 5
        assert config.continue_on_collection_error is True
        assert config.deduplicate_pairs is False
        assert config.max_pairs == 1000
        assert config.wait_for_completion is False

    def test_invalid_num_runs(self):
        """Test invalid num_runs raises error."""
        with pytest.raises(ValueError):
            NeMoCustomizerTrainerConfig(
                reward=RewardFunctionConfig(name="test_reward"),
                num_runs=0,
            )

    def test_invalid_max_pairs(self):
        """Test invalid max_pairs raises error."""
        with pytest.raises(ValueError):
            NeMoCustomizerTrainerConfig(
                reward=RewardFunctionConfig(name="test_reward"),
                max_pairs=0,
            )

    def test_config_name(self):
        """Test config is registered with correct name."""
        assert NeMoCustomizerTrainerConfig._typed_model_name == "nemo_customizer_trainer"


# =============================================================================
# Trainer Tests
# =============================================================================


class TestNeMoCustomizerTrainer:
    """Tests for NeMo Customizer Trainer."""

    @pytest.fixture
    def trainer_config(self):
        """Create test trainer configuration."""
        return NeMoCustomizerTrainerConfig(
            reward=RewardFunctionConfig(name="test_reward"),
            num_runs=3,
        )

    @pytest.fixture
    def finetune_config(self, tmp_path):
        """Create test finetune configuration."""
        config_file = tmp_path / "config.yml"
        config_file.write_text("test: config")

        dataset_file = tmp_path / "dataset.jsonl"
        dataset_file.write_text('{"input": "test"}')

        run_config = FinetuneRunConfig(
            config_file=config_file,
            target_functions=["test_function"],
            dataset=str(dataset_file),
            result_json_path="$.result",
        )

        return FinetuneConfig(
            run_configuration=run_config,
            reward_function=RewardFunctionConfig(name="test_reward"),
            output_dir=tmp_path / "output",
        )

    @pytest.fixture
    def trainer(self, trainer_config):
        """Create trainer instance."""
        return NeMoCustomizerTrainer(trainer_config=trainer_config)

    @pytest.fixture
    def sample_dpo_trajectories(self):
        """Create sample trajectories with DPO items."""
        dpo_item = DPOItem(
            prompt=[
                OpenAIMessage(role="system", content="You are helpful."),
                OpenAIMessage(role="user", content="What is 2+2?"),
            ],
            chosen_response="The answer is 4.",
            rejected_response="I don't know.",
        )

        trajectory = Trajectory(
            episode=[dpo_item],
            reward=0.5,
            metadata={"example_id": "ex_1"},
        )

        return [[trajectory]]

    def test_trainer_initialization(self, trainer, trainer_config):
        """Test that trainer initializes with correct configuration."""
        assert trainer.trainer_config == trainer_config
        assert trainer._job_ref is None
        assert trainer._run_id is None
        assert trainer._all_trajectories == []
        assert trainer._run_metrics == []

    async def test_trainer_initialize(self, trainer, finetune_config):
        """Test trainer initialization process."""
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)

        with patch.object(uuid, "uuid4", return_value=MagicMock(hex="abcd1234")):
            await trainer.initialize(finetune_config)

        assert trainer.run_config == finetune_config
        assert trainer._run_id.startswith("nemo_dpo_")
        assert trainer._run_id == "nemo_dpo_abcd1234"
        mock_builder.initialize.assert_called_once_with(finetune_config)
        mock_adapter.initialize.assert_called_once_with(finetune_config)

    async def test_trainer_initialize_no_curriculum(self, trainer, finetune_config):
        """Test curriculum learning is disabled for DPO."""
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        assert trainer.curriculum_config is None
        assert trainer._curriculum_state["current_percentile"] == 1.0

    async def test_run_epoch_collects_trajectories(self, trainer, finetune_config, sample_dpo_trajectories):
        """Test running epoch collects trajectories."""
        trajectory_collection = TrajectoryCollection(
            trajectories=sample_dpo_trajectories,
            run_id="test_run",
        )

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=trajectory_collection)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        result = await trainer.run_epoch(epoch=0, run_id="test_run")

        assert result is None  # No job submitted per-run
        assert len(trainer._all_trajectories) == 1
        assert len(trainer._run_metrics) == 1
        mock_builder.start_run.assert_called_once()
        mock_builder.finalize.assert_called_once()

    async def test_run_epoch_empty_trajectories(self, trainer, finetune_config):
        """Test running epoch with no trajectories."""
        empty_collection = TrajectoryCollection(
            trajectories=[],
            run_id="test_run",
        )

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=empty_collection)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        result = await trainer.run_epoch(epoch=0, run_id="test_run")

        assert result is None
        assert len(trainer._all_trajectories) == 0

    async def test_run_multiple_collection_runs(self, trainer, finetune_config, sample_dpo_trajectories):
        """Test running multiple data collection runs."""
        trajectory_collection = TrajectoryCollection(
            trajectories=sample_dpo_trajectories,
            run_id="test_run",
        )

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=trajectory_collection)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        mock_job_ref = TrainingJobRef(
            run_id="test_run",
            backend="nemo-customizer",
            metadata={"job_id": "job-123"},
        )
        mock_adapter.submit = AsyncMock(return_value=mock_job_ref)

        mock_status = TrainingJobStatus(
            run_id="test_run",
            backend="nemo-customizer",
            status=TrainingStatusEnum.COMPLETED,
        )
        mock_adapter.wait_until_complete = AsyncMock(return_value=mock_status)

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        statuses = await trainer.run(num_epochs=3)

        assert len(statuses) == 1
        assert statuses[0].status == TrainingStatusEnum.COMPLETED
        assert mock_builder.start_run.call_count == 3  # 3 runs
        assert mock_adapter.submit.call_count == 1  # Single submission

    async def test_run_no_wait_for_completion(self, trainer_config, finetune_config, sample_dpo_trajectories):
        """Test running without waiting for completion."""
        trainer_config.wait_for_completion = False
        trainer = NeMoCustomizerTrainer(trainer_config=trainer_config)

        trajectory_collection = TrajectoryCollection(
            trajectories=sample_dpo_trajectories,
            run_id="test_run",
        )

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=trajectory_collection)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        mock_job_ref = TrainingJobRef(
            run_id="test_run",
            backend="nemo-customizer",
            metadata={"job_id": "job-123"},
        )
        mock_adapter.submit = AsyncMock(return_value=mock_job_ref)

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        statuses = await trainer.run(num_epochs=1)

        assert len(statuses) == 1
        assert statuses[0].status == TrainingStatusEnum.RUNNING
        mock_adapter.wait_until_complete.assert_not_called()

    async def test_run_collection_error_stops(self, trainer, finetune_config):
        """Test collection error stops by default."""
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock(side_effect=Exception("Test error"))

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        statuses = await trainer.run(num_epochs=3)

        assert len(statuses) == 1
        assert statuses[0].status == TrainingStatusEnum.FAILED
        assert "Test error" in statuses[0].message

    async def test_run_collection_error_continues(self, trainer_config, finetune_config, sample_dpo_trajectories):
        """Test collection error continues when configured."""
        trainer_config.continue_on_collection_error = True
        trainer_config.num_runs = 3
        trainer = NeMoCustomizerTrainer(trainer_config=trainer_config)

        trajectory_collection = TrajectoryCollection(
            trajectories=sample_dpo_trajectories,
            run_id="test_run",
        )

        call_count = [0]

        async def finalize_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Run 2 failed")
            return trajectory_collection

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(side_effect=finalize_side_effect)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        mock_job_ref = TrainingJobRef(
            run_id="test_run",
            backend="nemo-customizer",
            metadata={"job_id": "job-123"},
        )
        mock_adapter.submit = AsyncMock(return_value=mock_job_ref)

        mock_status = TrainingJobStatus(
            run_id="test_run",
            backend="nemo-customizer",
            status=TrainingStatusEnum.COMPLETED,
        )
        mock_adapter.wait_until_complete = AsyncMock(return_value=mock_status)

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        statuses = await trainer.run(num_epochs=3)

        # Should complete despite error in run 2
        assert statuses[0].status == TrainingStatusEnum.COMPLETED
        assert mock_builder.start_run.call_count == 3

    async def test_run_no_trajectories_fails(self, trainer, finetune_config):
        """Test run fails when no trajectories collected."""
        empty_collection = TrajectoryCollection(
            trajectories=[],
            run_id="test_run",
        )

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.start_run = AsyncMock()
        mock_builder.finalize = AsyncMock(return_value=empty_collection)

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        statuses = await trainer.run(num_epochs=3)

        assert len(statuses) == 1
        assert statuses[0].status == TrainingStatusEnum.FAILED
        assert "No trajectories collected" in statuses[0].message

    def test_deduplicate_trajectories(self, trainer, sample_dpo_trajectories):
        """Test trajectory deduplication."""
        # Create duplicate trajectories
        dpo_item1 = DPOItem(
            prompt="Same prompt",
            chosen_response="Same chosen",
            rejected_response="Same rejected",
        )
        dpo_item2 = DPOItem(
            prompt="Same prompt",
            chosen_response="Same chosen",
            rejected_response="Same rejected",
        )
        dpo_item3 = DPOItem(
            prompt="Different prompt",
            chosen_response="Different chosen",
            rejected_response="Different rejected",
        )

        trajectories = [
            [Trajectory(episode=[dpo_item1], reward=0.5, metadata={})],
            [Trajectory(episode=[dpo_item2], reward=0.5, metadata={})],
            [Trajectory(episode=[dpo_item3], reward=0.7, metadata={})],
        ]

        collection = TrajectoryCollection(
            trajectories=trajectories,
            run_id="test_run",
        )

        result = trainer._deduplicate_trajectories(collection)

        # Should remove duplicate
        assert len(result.trajectories) == 2

    def test_sample_trajectories(self, trainer):
        """Test trajectory sampling."""
        trajectories = [[
            Trajectory(
                episode=[DPOItem(
                    prompt=f"prompt_{i}",
                    chosen_response="chosen",
                    rejected_response="rejected",
                )],
                reward=0.5,
                metadata={},
            )
        ] for i in range(10)]

        collection = TrajectoryCollection(
            trajectories=trajectories,
            run_id="test_run",
        )

        result = trainer._sample_trajectories(collection, max_pairs=5)

        assert len(result.trajectories) == 5

    def test_sample_trajectories_below_limit(self, trainer):
        """Test sampling returns unchanged when below limit."""
        trajectories = [[
            Trajectory(
                episode=[DPOItem(
                    prompt=f"prompt_{i}",
                    chosen_response="chosen",
                    rejected_response="rejected",
                )],
                reward=0.5,
                metadata={},
            )
        ] for i in range(3)]

        collection = TrajectoryCollection(
            trajectories=trajectories,
            run_id="test_run",
        )

        result = trainer._sample_trajectories(collection, max_pairs=10)

        assert result == collection

    async def test_get_metrics(self, trainer, finetune_config):
        """Test getting metrics."""
        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        mock_status = TrainingJobStatus(
            run_id="test_run",
            backend="nemo-customizer",
            status=TrainingStatusEnum.RUNNING,
            progress=50.0,
        )
        mock_adapter.status = AsyncMock(return_value=mock_status)

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        trainer._run_metrics = [
            {
                "run_number": 0, "num_trajectories": 10, "num_dpo_pairs": 20
            },
            {
                "run_number": 1, "num_trajectories": 15, "num_dpo_pairs": 30
            },
        ]
        trainer._job_ref = TrainingJobRef(
            run_id="test_run",
            backend="nemo-customizer",
            metadata={"job_id": "job-123"},
        )

        metrics = await trainer.get_metrics("test_run")

        assert metrics["run_id"] == "test_run"
        assert metrics["num_collection_runs"] == 2
        assert len(metrics["collection_runs"]) == 2
        assert metrics["training_job"]["status"] == "running"

    def test_log_progress(self, trainer, finetune_config, tmp_path):
        """Test logging progress to file."""
        trainer.run_config = finetune_config
        trainer._run_id = "test_run"

        metrics = {
            "num_trajectories": 10,
            "num_dpo_pairs": 20,
            "avg_reward": 0.75,
        }

        trainer.log_progress(epoch=0, metrics=metrics, output_dir=str(tmp_path))

        assert (tmp_path / "data_collection_progress.jsonl").exists()
        assert (tmp_path / "collection_history.json").exists()

        with open(tmp_path / "data_collection_progress.jsonl") as f:
            log_entry = json.loads(f.readline())
            assert log_entry["run_number"] == 0
            assert log_entry["num_dpo_pairs"] == 20

    async def test_cleanup(self, trainer, finetune_config):
        """Test cleanup clears data."""
        eval_task = MagicMock()
        eval_task.done.return_value = False
        eval_task.cancel = MagicMock()

        mock_builder = MagicMock()
        mock_builder.initialize = AsyncMock()
        mock_builder.evaluation_runs = {"run1": eval_task}

        mock_adapter = MagicMock()
        mock_adapter.initialize = AsyncMock()

        await trainer.bind_components(mock_builder, mock_adapter)
        await trainer.initialize(finetune_config)

        trainer._all_trajectories = [[MagicMock()]]
        trainer._run_metrics = [{"test": "data"}]

        await trainer.cleanup()

        assert trainer._all_trajectories == []
        assert trainer._run_metrics == []
        eval_task.cancel.assert_called_once()

    async def test_run_not_initialized_raises(self, trainer):
        """Test run raises error if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await trainer.run(num_epochs=1)
