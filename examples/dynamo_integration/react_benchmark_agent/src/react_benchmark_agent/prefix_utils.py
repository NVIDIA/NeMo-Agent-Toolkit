# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Managing Dynamo prefix IDs for evaluation questions.

This module shows how to use the NAT LangChain plugin's prefix ID management
to ensure all LLM calls within a single evaluation question share the same
prefix ID for optimal Dynamo KV cache reuse.
"""

import logging
import uuid
from typing import Any

# Import the core functionality from NAT's LangChain plugin
from nat.plugins.langchain.dynamo_prefix_headers import (
    set_prefix_id_for_question,
    clear_prefix_id,
    get_current_prefix_id,
)

logger = logging.getLogger(__name__)


def generate_benchmark_prefix(question_id: str, template: str = "react-benchmark-{question_id}-{uuid}") -> str:
    """
    Generate a unique prefix ID for a benchmark question.
    
    Args:
        question_id: Question identifier (for example, "banking_scenario_001")
        template: Template with {question_id} and {uuid} placeholders
        
    Returns:
        Unique prefix ID for this question
        
    Example:
        >>> generate_benchmark_prefix("banking_scenario_001")
        'react-benchmark-banking_scenario_001-a1b2c3d4'
    """
    unique_suffix = uuid.uuid4().hex[:8]
    return template.format(question_id=question_id, uuid=unique_suffix)


async def run_question_with_prefix(
    question_id: str,
    question_text: str,
    workflow_fn: Any,
    **workflow_kwargs
) -> Any:
    """
    Run a single evaluation question with a unique prefix ID.
    
    This ensures all LLM calls for this question share the same prefix ID,
    enabling Dynamo to reuse KV cache across the multi-turn conversation.
    
    Args:
        question_id: Unique identifier for the question
        question_text: The actual question text
        workflow_fn: The workflow function to execute
        **workflow_kwargs: Additional arguments for the workflow
        
    Returns:
        The workflow result
        
    Example:
        >>> from nat.builder.workflow_builder import WorkflowBuilder
        >>> async with WorkflowBuilder.from_config("config.yml") as builder:
        >>>     workflow = await builder.get_workflow()
        >>>     result = await run_question_with_prefix(
        >>>         "q001",
        >>>         "What is my account balance?",
        >>>         workflow.ainvoke
        >>>     )
    """
    # Generate unique prefix for this question
    prefix_id = generate_benchmark_prefix(question_id)
    
    # Set it in the context
    set_prefix_id_for_question(prefix_id)
    
    logger.info("Running question %s with prefix ID: %s", question_id, prefix_id)
    
    try:
        # Run the workflow - all LLM calls will use this prefix ID
        result = await workflow_fn(question_text, **workflow_kwargs)
        logger.debug("Question %s completed successfully", question_id)
        return result
        
    finally:
        # Clean up the context
        clear_prefix_id()
        logger.debug("Cleared prefix ID for question %s", question_id)


# Example: Manual eval loop with prefix management
async def example_manual_eval_loop(workflow, questions: list[tuple[str, str]]):
    """
    Example of manually managing prefix IDs in an evaluation loop.
    
    Args:
        workflow: NAT workflow instance
        questions: List of (question_id, question_text) tuples
    """
    results = []
    
    for question_id, question_text in questions:
        # Each question gets a unique prefix
        result = await run_question_with_prefix(
            question_id=question_id,
            question_text=question_text,
            workflow_fn=workflow.ainvoke,
        )
        results.append(result)
    
    return results


# Example: Inline usage for quick tests
async def example_inline_usage():
    """Example of direct inline usage."""
    from nat.builder.workflow_builder import WorkflowBuilder
    
    async with WorkflowBuilder.from_config("config.yml") as builder:
        workflow = await builder.get_workflow()
        
        # Question 1
        set_prefix_id_for_question("react-benchmark-q001-abc123")
        result1 = await workflow.ainvoke("Transfer $500 to savings")
        clear_prefix_id()
        
        # Question 2  
        set_prefix_id_for_question("react-benchmark-q002-def456")
        result2 = await workflow.ainvoke("Check my account balance")
        clear_prefix_id()
        
        return result1, result2

