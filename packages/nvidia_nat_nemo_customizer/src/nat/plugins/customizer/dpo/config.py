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
Configuration classes for DPO training with NeMo Customizer.

This module provides configuration for:
1. DPO Trajectory Builder - collecting preference data from workflows
2. NeMo Customizer TrainerAdapter - submitting DPO training jobs
"""

from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.data_models.finetuning import TrainerAdapterConfig
from nat.data_models.finetuning import TrainerConfig
from nat.data_models.finetuning import TrajectoryBuilderConfig


class DPOTrajectoryBuilderConfig(TrajectoryBuilderConfig, name="dpo_traj_builder"):
    """
    Configuration for the DPO (Direct Preference Optimization) Trajectory Builder.

    This builder collects preference pairs from workflows that produce TTC_END
    intermediate steps with TTCEventData. It uses the structured TTCEventData
    model to extract turn_id, candidate_index, score, input (prompt), and
    output (response) - no dictionary key configuration needed.

    The builder groups candidates by turn_id and creates preference pairs based
    on score differences.

    Example YAML configuration::

        trajectory_builders:
          dpo_builder:
            _type: dpo_traj_builder
            ttc_step_name: dpo_candidate_move
            exhaustive_pairs: true
            min_score_diff: 0.05
            max_pairs_per_turn: 5
    """

    # === Step Filtering ===
    ttc_step_name: str = Field(
        default="dpo_candidate_move",
        description="Name of the TTC intermediate step to collect. "
        "The builder filters for TTC_END events with this name.",
    )

    # === Pair Generation Modes ===
    exhaustive_pairs: bool = Field(
        default=True,
        description="If True, generate all pairwise comparisons where "
        "score(A) > score(B). If False, only generate best vs worst pair.",
    )

    min_score_diff: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum score difference required to create a preference "
        "pair. Pairs with smaller differences are filtered out.",
    )

    max_pairs_per_turn: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of preference pairs to generate per turn. "
        "If None, no limit. Pairs sorted by score difference (highest first).",
    )

    # === Reward Computation ===
    reward_from_score_diff: bool = Field(
        default=True,
        description="If True, compute trajectory reward as score difference "
        "(chosen - rejected). If False, use chosen score directly as reward.",
    )

    # === Validation ===
    require_multiple_candidates: bool = Field(
        default=True,
        description="If True, skip turns with only one candidate (no preference "
        "signal). If False, include single-candidate turns.",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "DPOTrajectoryBuilderConfig":
        """Validate configuration consistency."""
        if self.max_pairs_per_turn is not None and self.max_pairs_per_turn < 1:
            raise ValueError("max_pairs_per_turn must be at least 1 if specified")
        return self


# =============================================================================
# NeMo Customizer Trainer Configuration
# =============================================================================


class NeMoCustomizerTrainerConfig(TrainerConfig, name="nemo_customizer_trainer"):
    """
    Configuration for the NeMo Customizer Trainer.

    This trainer orchestrates DPO data collection and training job submission.
    Unlike epoch-based trainers, it runs the trajectory builder multiple times
    to collect data, then submits a single training job to NeMo Customizer.

    Example YAML configuration::

        trainers:
          nemo_dpo:
            _type: nemo_customizer_trainer
            num_runs: 5
            wait_for_completion: true
            deduplicate_pairs: true
            max_pairs: 10000
    """

    # === Data Collection ===
    num_runs: int = Field(
        default=1,
        ge=1,
        description="Number of times to run the trajectory builder to collect data. "
        "Each run generates preference pairs from the evaluation dataset. "
        "Multiple runs can increase dataset diversity.",
    )

    continue_on_collection_error: bool = Field(
        default=False,
        description="If True, continue with remaining runs if one fails. "
        "If False, stop immediately on first error.",
    )

    # === Data Processing ===
    deduplicate_pairs: bool = Field(
        default=True,
        description="If True, remove duplicate DPO pairs based on prompt+responses. "
        "Useful when multiple runs may generate the same pairs.",
    )

    max_pairs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of DPO pairs to include in training. "
        "If None, use all collected pairs. If set, randomly samples pairs.",
    )

    # === Training Job ===
    wait_for_completion: bool = Field(
        default=True,
        description="If True, wait for the NeMo Customizer training job to complete. "
        "If False, submit the job and return immediately.",
    )


# =============================================================================
# NeMo Customizer TrainerAdapter Configuration
# =============================================================================


class DPOSpecificHyperparameters(BaseModel):
    """DPO-specific hyperparameters for NeMo Customizer."""

    ref_policy_kl_penalty: float = Field(
        default=0.1,
        ge=0.0,
        description="KL penalty coefficient for reference policy regularization.",
    )

    preference_loss_weight: float = Field(default=1.0,
                                          ge=0.0,
                                          description="Scales the contribution of the preference loss")

    preference_average_log_probs: bool = Field(
        default=False,
        description="If True, use average log probabilities over sequence length "
        "when computing preference loss. If False, use sum of log probabilities.",
    )

    sft_loss_weight: float = Field(default=0.0,
                                   ge=0.0,
                                   description="Scales the contribution of the supervised fine-tuning (SFT) loss. ")


class NeMoCustomizerHyperparameters(BaseModel):
    """
    Hyperparameters for NeMo Customizer training jobs.

    These map to the `hyperparameters` argument in
    `client.customization.jobs.create()`.
    """

    training_type: Literal["sft", "dpo"] = Field(
        default="dpo",
        description="Type of training: 'sft' for supervised fine-tuning, 'dpo' for direct preference optimization.",
    )
    finetuning_type: Literal["lora", "all_weights"] = Field(
        default="all_weights",
        description="Type of finetuning: 'lora' for LoRA adapters, 'all_weights' for full model.",
    )
    epochs: int = Field(
        default=3,
        ge=1,
        description="Number of training epochs.",
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        description="Training batch size.",
    )
    learning_rate: float = Field(
        default=5e-5,
        gt=0.0,
        description="Learning rate for optimizer.",
    )
    dpo: DPOSpecificHyperparameters = Field(
        default_factory=DPOSpecificHyperparameters,
        description="DPO-specific hyperparameters.",
    )


class NIMDeploymentConfig(BaseModel):
    """
    Configuration for NIM deployment after training.

    These settings are used when `deploy_on_completion` is True.
    """

    image_name: str = Field(
        default="nvcr.io/nim/meta/llama-3.1-8b-instruct",
        description="NIM container image name.",
    )
    image_tag: str = Field(
        default="latest",
        description="NIM container image tag.",
    )
    gpu: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for deployment.",
    )
    deployment_name: str | None = Field(
        default=None,
        description="Name for the deployment. If None, auto-generated from model name.",
    )
    description: str = Field(
        default="Fine-tuned model deployment",
        description="Description for the deployment.",
    )


class NeMoCustomizerTrainerAdapterConfig(TrainerAdapterConfig, name="nemo_customizer_trainer_adapter"):
    """
    Configuration for the NeMo Customizer TrainerAdapter.

    This adapter submits DPO/SFT training jobs to NeMo Customizer and
    optionally deploys the trained model.

    Example YAML configuration::

        trainer_adapters:
          nemo_customizer:
            _type: nemo_customizer_trainer_adapter
            entity_host: https://nmp.example.com
            datastore_host: https://datastore.example.com
            namespace: my-project
            customization_config: meta/llama-3.2-1b-instruct@v1.0.0+A100
            hyperparameters:
              training_type: dpo
              epochs: 5
              batch_size: 8
            use_full_message_history: true
            deploy_on_completion: true
    """

    # === Endpoint Configuration ===
    entity_host: str = Field(description="Base URL for NeMo Entity Store (e.g., https://nmp.example.com).", )
    datastore_host: str = Field(description="Base URL for NeMo Datastore (e.g., https://datastore.example.com).", )
    hf_token: str = Field(
        default="",
        description="HuggingFace token for datastore authentication. Can be empty if not required.",
    )

    # === Namespace and Dataset ===
    namespace: str = Field(description="Namespace for organizing resources (datasets, models, deployments).", )
    dataset_name: str = Field(
        default="nat-dpo",
        description="Name for the training dataset. Must be unique within namespace.",
    )
    dataset_output_dir: str | None = Field(
        default=None,
        description="Directory to save dataset JSONL files locally before upload. "
        "If None, uses a temporary directory that is deleted after upload. "
        "If specified, creates the directory if it doesn't exist and preserves files.",
    )
    create_namespace_if_missing: bool = Field(
        default=True,
        description="If True, create namespace in entity store and datastore if it doesn't exist.",
    )

    # === Customization Job Configuration ===
    customization_config: str = Field(description="Model configuration string for customization job "
                                      "(e.g., 'meta/llama-3.2-1b-instruct@v1.0.0+A100'). "
                                      "Available configs can be listed via the NeMo Customizer API.", )
    hyperparameters: NeMoCustomizerHyperparameters = Field(
        default_factory=NeMoCustomizerHyperparameters,
        description="Hyperparameters for the training job.",
    )

    # === Prompt Formatting ===
    use_full_message_history: bool = Field(
        default=False,
        description="If True, include full message history in prompt field as list of messages. "
        "If False, use only the last message content as a string. "
        "Full history format: [{\"role\": \"system\", \"content\": \"...\"}, ...]. "
        "Last message format: \"<content string>\".",
    )

    # === Deployment Configuration ===
    deploy_on_completion: bool = Field(
        default=False,
        description="If True, automatically deploy the trained model after job completion.",
    )
    deployment_config: NIMDeploymentConfig = Field(
        default_factory=NIMDeploymentConfig,
        description="Configuration for model deployment (used when deploy_on_completion=True).",
    )

    # === Polling Configuration ===
    poll_interval_seconds: float = Field(
        default=30.0,
        gt=0.0,
        description="Interval in seconds between job status checks.",
    )
    deployment_timeout_seconds: float = Field(
        default=1800.0,
        gt=0.0,
        description="Maximum time in seconds to wait for deployment to be ready. "
        "Default is 30 minutes (1800 seconds).",
    )

    @model_validator(mode="after")
    def validate_config(self) -> "NeMoCustomizerTrainerAdapterConfig":
        """Validate configuration consistency."""
        # Ensure hosts don't have trailing slashes
        self.entity_host = self.entity_host.rstrip("/")
        self.datastore_host = self.datastore_host.rstrip("/")
        return self
