# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base Defense Middleware.

This module provides a utility base class for defense middleware with common
configuration and helper methods. Each defense middleware implements its own
core logic based on its specific defense strategy (LLM-based, rule-based, etc.).
"""

import logging
from typing import Any, Optional, Union

from pydantic import Field

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.data_models.middleware import FunctionMiddlewareBaseConfig
from nat.middleware.function_middleware import FunctionMiddleware

logger = logging.getLogger(__name__)


class DefenseMiddlewareConfig(FunctionMiddlewareBaseConfig):
    """Base configuration for defense middleware.
    
    Provides common configuration fields that most defense middleware share.
    Not all fields are required - subclasses can override or add their own.
    """
    
    action: str = Field(
        default="log",
        description="Action to take when threat detected: 'log', 'block', or 'sanitize'"
    )
    
    check_input: bool = Field(
        default=False,
        description="Check function input for threats"
    )
    
    check_output: bool = Field(
        default=True,
        description="Check function output for threats"
    )
    
    llm_wrapper_type: Union[LLMFrameworkEnum, str] = Field(
        default=LLMFrameworkEnum.LANGCHAIN,
        description="Framework wrapper type for LLM (langchain, llama_index, crewai, etc.). "
                    "Only needed for LLM-based defenses."
    )
    
    target_function_or_group: Optional[str] = Field(
        default=None,
        description="Optional function or function group to target. "
                    "If None, defense applies to all functions. "
                    "Examples: 'my_calculator', 'my_calculator.divide', 'llm_agent.generate'"
    )


class DefenseMiddleware(FunctionMiddleware):
    """Utility base class for defense middleware.
    
    This base class provides:
    - Common configuration fields (action, check_input, check_output, llm_wrapper_type)
    - Helper methods for LLM loading (for LLM-based defenses)
    - Access to builder for any resources needed
    
    Unlike an abstract base class, this does NOT enforce a specific pattern.
    Each defense middleware implements its own invoke/stream logic based on
    its specific defense strategy:
    - LLM-based analysis (guard models, verifiers)
    - Rule-based detection (regex, signatures)
    - Heuristic-based checks
    - Statistical anomaly detection
    - etc.
    
    Each defense owns its core logic, just like red_teaming_middleware does.
    
    LLM Wrapper Types:
        The llm_wrapper_type config field supports different framework wrappers:
        - langchain (default) - For LangChain/LangGraph-based workflows
        - llama_index - For LlamaIndex-based workflows
        - crewai - For CrewAI-based workflows
        - semantic_kernel - For Semantic Kernel-based workflows
        - agno, adk, strands - Other supported frameworks
    """
    
    def __init__(self, config: DefenseMiddlewareConfig, builder):
        """Initialize defense middleware.
        
        Args:
            config: Configuration for the defense middleware
            builder: Builder instance for loading LLMs and other resources
        """
        super().__init__(is_final=False)
        self.config = config
        self.builder = builder
        
        logger.info(
            f"{self.__class__.__name__} initialized: "
            f"action={config.action}, check_input={config.check_input}, "
            f"check_output={config.check_output}, target={config.target_function_or_group}"
        )
    
    def _should_apply_defense(self, context_name: str) -> bool:
        """Check if defense should be applied to this function based on targeting configuration.
        
        This method mirrors the targeting logic from RedTeamingMiddleware to provide
        consistent behavior between attack and defense middleware.
        
        Args:
            context_name: The name of the function from context (e.g., "calculator.add")
        
        Returns:
            True if defense should be applied, False otherwise
        
        Examples:
            - target=None → defends all functions
            - target="my_calculator" → defends all functions in my_calculator group
            - target="my_calculator.divide" → defends only the divide function
        """
        # If no target specified, defend all functions
        if self.config.target_function_or_group is None:
            return True
        
        target = self.config.target_function_or_group
        
        # Group targeting - match if context starts with the group name
        # Handle both "group.function" and just "function" in context
        if "." in context_name and "." not in target:
            context_group = context_name.split(".", 1)[0]
            return context_group == target
        
        # Exact match for specific function or group
        return context_name == target
    
    async def _get_llm_for_defense(self, llm_name: str, wrapper_type: Union[LLMFrameworkEnum, str, None] = None):
        """Helper to lazy load an LLM for defense purposes.
        
        This is a utility method for LLM-based defenses. Not all defenses
        will use this - some may use rule-based or other detection methods.
        
        Args:
            llm_name: Name of the LLM to load
            wrapper_type: Framework wrapper type (defaults to config.llm_wrapper_type if not specified)
        
        Returns:
            The loaded LLM instance with the specified framework wrapper
        """
        if wrapper_type is None:
            wrapper_type = self.config.llm_wrapper_type
        
        return await self.builder.get_llm(
            llm_name, 
            wrapper_type=wrapper_type
        )


__all__ = ["DefenseMiddleware", "DefenseMiddlewareConfig"]
