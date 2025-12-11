# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Custom Dynamo-aware LLM config with optimizable prefix parameters.

This extends the OpenAI LLM config to add proper schema definitions for
Dynamo prefix parameters, making them discoverable by the NAT optimizer.
"""

from typing import Literal

from pydantic import ConfigDict

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.llm import LLMProviderInfo
from nat.cli.register_workflow import register_llm_client, register_llm_provider
from nat.data_models.optimizable import OptimizableField, SearchSpace
from nat.llm.openai_llm import OpenAIModelConfig


# Define valid prefix hint values
PrefixLevel = Literal["LOW", "MEDIUM", "HIGH"]


class DynamoLLMConfig(OpenAIModelConfig, name="dynamo_openai"):
    """
    OpenAI-compatible LLM config with Dynamo prefix optimization support.

    This config extends the standard OpenAI config to include proper schema
    definitions for Dynamo router prefix parameters, enabling them to be
    discovered and optimized by the NAT optimizer.

    Dynamo Prefix Parameters:
    -------------------------
    - prefix_osl (Output Sequence Length): Hint for expected response length
        - LOW: decode_cost=1.0, short responses
        - MEDIUM: decode_cost=2.0, typical responses
        - HIGH: decode_cost=3.0, long responses

    - prefix_iat (Inter-Arrival Time): Hint for request pacing
        - LOW: iat_factor=1.5, rapid bursts → high worker stickiness
        - MEDIUM: iat_factor=1.0, normal pacing
        - HIGH: iat_factor=0.6, slow requests → more exploration

    - prefix_total_requests: Expected requests per conversation
        - Higher values increase KV cache affinity and worker stickiness
        - Lower values allow more load balancing
    """

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    # =========================================================================
    # DYNAMO PREFIX PARAMETERS - OPTIMIZABLE
    # =========================================================================

    enable_dynamic_prefix: bool = OptimizableField(
        default=True,
        description="Enable dynamic prefix ID generation for KV cache optimization."
    )

    prefix_template: str = OptimizableField(
        default="nat-dynamo-{uuid}",
        description="Template for generating prefix IDs. Use {uuid} for unique ID."
    )

    prefix_total_requests: int = OptimizableField(
        default=10,
        ge=1,
        le=50,
        description=(
            "Expected number of requests for this conversation/prefix. "
            "Higher values increase worker stickiness and KV cache locality. "
            "Lower values allow more load balancing across workers."
        ),
        space=SearchSpace(low=1, high=20, step=5)
    )

    prefix_osl: PrefixLevel = OptimizableField(
        default="MEDIUM",
        description=(
            "Output Sequence Length hint for the Dynamo router. "
            "LOW=short responses (decode_cost=1.0), "
            "MEDIUM=typical (decode_cost=2.0), "
            "HIGH=long responses (decode_cost=3.0)."
        ),
        space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"])
    )

    prefix_iat: PrefixLevel = OptimizableField(
        default="MEDIUM",
        description=(
            "Inter-Arrival Time hint for the Dynamo router. "
            "LOW=rapid bursts (iat_factor=1.5, high stickiness), "
            "MEDIUM=normal (iat_factor=1.0), "
            "HIGH=slow requests (iat_factor=0.6, more exploration)."
        ),
        space=SearchSpace(values=["LOW", "MEDIUM", "HIGH"])
    )


@register_llm_provider(config_type=DynamoLLMConfig)
async def dynamo_openai_llm(config: DynamoLLMConfig, _builder: Builder):
    """
    Register the Dynamo-aware OpenAI LLM provider.

    This provider is functionally identical to the standard OpenAI provider,
    but includes optimizable prefix parameters for Dynamo router tuning.
    """
    yield LLMProviderInfo(
        config=config,
        description="OpenAI-compatible LLM with Dynamo prefix optimization support."
    )


@register_llm_client(config_type=DynamoLLMConfig, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
async def dynamo_openai_langchain(llm_config: DynamoLLMConfig, _builder: Builder):
    """
    Create a LangChain ChatOpenAI client from DynamoLLMConfig.

    This is functionally identical to the OpenAI LangChain client but also
    passes the Dynamo prefix parameters as default headers for KV cache optimization.
    """
    from langchain_openai import ChatOpenAI

    from nat.data_models.llm import APITypeEnum

    # Build config dict excluding prefix-specific and NAT-specific fields
    config_dict = llm_config.model_dump(
        exclude={"type", "thinking", "api_type", "enable_dynamic_prefix",
                 "prefix_template", "prefix_total_requests", "prefix_osl", "prefix_iat"},
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )

    # Create the ChatOpenAI client
    if llm_config.api_type == APITypeEnum.RESPONSES:
        client = ChatOpenAI(
            stream_usage=True,
            use_responses_api=True,
            use_previous_response_id=True,
            **config_dict
        )
    else:
        client = ChatOpenAI(stream_usage=True, **config_dict)

    # Apply dynamic prefix headers for Dynamo if enabled
    if llm_config.enable_dynamic_prefix:
        from nat.plugins.langchain.dynamo_prefix_headers import patch_with_dynamic_prefix_headers
        client = patch_with_dynamic_prefix_headers(
            client,
            enable_dynamic_prefix=True,
            prefix_template=llm_config.prefix_template,
            total_requests=llm_config.prefix_total_requests,
            osl=llm_config.prefix_osl,
            iat=llm_config.prefix_iat,
        )

    yield client
