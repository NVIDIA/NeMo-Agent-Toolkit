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
"""Tests for NeMo Customizer TrainerAdapter."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.data_models.finetuning import DPOItem
from nat.data_models.finetuning import OpenAIMessage
from nat.data_models.finetuning import TrainingJobRef
from nat.data_models.finetuning import TrainingStatusEnum
from nat.data_models.finetuning import Trajectory
from nat.data_models.finetuning import TrajectoryCollection
from nat.plugins.customizer.dpo.config import DPOSpecificHyperparameters
from nat.plugins.customizer.dpo.config import NeMoCustomizerHyperparameters
from nat.plugins.customizer.dpo.config import NeMoCustomizerTrainerAdapterConfig
from nat.plugins.customizer.dpo.config import NIMDeploymentConfig
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

        assert config.image_name == "nvcr.io/nim/meta/llama-3.2-1b-instruct"
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
        assert config.use_full_message_history is True
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
        """Test health check failure."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = Exception("Connection refused")

            result = await trainer_adapter.is_healthy()

            assert result is False

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
