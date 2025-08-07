# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Guardrails provider registration."""

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_guardrails
from aiq.guardrails.interface import GuardrailsProviderFactory

from .config import NemoGuardrailsConfig
from .provider import NemoGuardrailsProvider

# Register the NeMo provider in the factory
GuardrailsProviderFactory.register_provider(NemoGuardrailsConfig, NemoGuardrailsProvider)


@register_guardrails(config_type=NemoGuardrailsConfig)
async def nemo_guardrails(guardrails_config: NemoGuardrailsConfig, builder: Builder):
    """Create NeMo Guardrails provider."""
    # Get LLM config if specified
    llm_config = None
    if hasattr(guardrails_config, 'llm_name') and guardrails_config.llm_name:
        llm_config = builder.get_llm_config(guardrails_config.llm_name)

    # Create and initialize provider
    provider = NemoGuardrailsProvider(guardrails_config, llm_config)
    await provider.initialize()
    yield provider
