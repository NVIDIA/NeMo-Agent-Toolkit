# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from aiq.data_models.guardrails import GuardrailsBaseConfig


class NemoGuardrailsConfig(GuardrailsBaseConfig, name="nemo_guardrails"):
    """Configuration for NeMo Guardrails integration."""

    enabled: bool = Field(default=True, description="Whether guardrails are enabled")

    input_rails_enabled: bool = Field(default=True, description="Whether input guardrails are enabled")

    output_rails_enabled: bool = Field(default=True, description="Whether output guardrails are enabled")

    config_path: str | None = Field(
        default=None,
        description="Path to the NeMo Guardrails configuration directory. If not specified, uses default configuration."
    )

    llm_name: str | None = Field(default=None,
                                 description="Name of the LLM to use for guardrails (references llms config)")

    fallback_response: str | None = Field(default="I cannot provide a response to that request.",
                                          description="Fallback response when guardrails are triggered")

    fallback_on_error: bool = Field(default=True,
                                    description="Whether to use fallback response when guardrails encounter errors")

    verbose: bool = Field(default=False, description="Whether to enable verbose logging for guardrails")

    max_retries: int = Field(default=3, description="Maximum number of retries when guardrails fail")

    timeout_seconds: float | None = Field(default=30.0, description="Timeout for guardrails processing in seconds")

    # Keep the original rails structure - supports both simple strings and detailed configs
    rails: dict[str, Any] | None = Field(
        default=None, description="Rails configuration (e.g., {'input': {'flows': ['self check input']}})")
