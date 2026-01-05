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
"""
Self-Evaluating Agent Wrapper with Optional Feedback Loop.

This module provides a wrapper around the ReAct agent that adds self-evaluation
and retry capabilities. After the agent completes its reasoning, it evaluates
whether the tool call chain is sufficient for the input question.

Two configuration modes are supported:
1. Basic mode (pass_feedback_to_agent=False): Retries without feedback
2. Advanced mode (pass_feedback_to_agent=True): Passes evaluation feedback to agent on retry

Both modes are registered:
- `self_evaluating_agent` - Legacy name, defaults to no feedback (backward compatible)
- `self_evaluating_agent_with_feedback` - Advanced mode with feedback enabled by default
"""

import hashlib
import json
import logging
from typing import Any

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig

# Import global intent functions for cross-builder access
from .tool_intent_stubs import clear_global_intents
from .tool_intent_stubs import get_global_intents
from .tool_intent_stubs import set_current_scenario_id

logger = logging.getLogger(__name__)


def _get_scenario_id_for_question(question: str) -> str:
    """Generate a unique scenario ID for a question using content hash."""
    # Use question hash for uniqueness (contextvars handle async isolation)
    question_hash = hashlib.md5(question.encode()).hexdigest()[:12]
    return f"q_{question_hash}"


# =============================================================================
# LEGACY CONFIG: self_evaluating_agent (backward compatible)
# =============================================================================
class SelfEvaluatingAgentConfig(FunctionBaseConfig, name="self_evaluating_agent"):
    """
    Configuration for the Self-Evaluating Agent (legacy mode without feedback).

    This agent wraps another agent (typically ReAct) and adds self-evaluation
    and retry capabilities. This is the backward-compatible configuration that
    does NOT pass feedback to the agent on retry.

    For the advanced version with feedback, use `self_evaluating_agent_with_feedback`.
    """

    wrapped_agent: FunctionRef = Field(
        ..., description="The underlying agent to wrap (e.g., react_agent with decision_only mode)")
    evaluator_llm: LLMRef = Field(..., description="LLM to use for self-evaluation")
    max_retries: int = Field(default=2, description="Maximum number of retry attempts", ge=0, le=5)
    min_confidence_threshold: float = Field(default=0.7,
                                            description="Minimum confidence to accept the tool sequence",
                                            ge=0.0,
                                            le=1.0)
    pass_feedback_to_agent: bool = Field(default=False,
                                         description="Whether to pass evaluation feedback to the agent on retry")
    feedback_template: str = Field(
        default="""PREVIOUS ATTEMPT FEEDBACK:

Your previous tool selection was evaluated and found to be insufficient.

EVALUATION:
{reasoning}

MISSING STEPS:
{missing_steps}

SUGGESTIONS:
{suggestions}

Please try again, addressing the issues identified above. Focus on:
1. Including all necessary information gathering steps
2. Ensuring proper order of operations
3. Adding verification steps where appropriate
""",
        description="Template for feedback passed to agent on retry (only used if pass_feedback_to_agent=True)",
    )
    evaluation_prompt_template: str = Field(
        default="""You are evaluating whether a sequence of tool calls is sufficient to answer a user question.

USER QUESTION:
{question}

TOOL CALLS MADE:
{tool_calls}

EVALUATION CRITERIA:
1. Do the tool calls logically address the user's question?
2. Are all necessary information gathering steps included?
3. Are the tool calls in a reasonable order?
4. Are there any missing critical steps?
5. Are there any redundant or unnecessary tool calls?

Based on these criteria, is this tool call sequence SUFFICIENT to answer the user's question?

Respond with a JSON object:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your evaluation",
    "missing_steps": ["list", "of", "missing", "steps"] or [],
    "suggestions": "suggestions for improvement if insufficient"
}}

JSON Response:""",
        description="Template for self-evaluation prompt. Available variables: {question}, {tool_calls}",
    )
    verbose: bool = Field(default=True, description="Enable verbose logging")


# =============================================================================
# ADVANCED CONFIG: self_evaluating_agent_with_feedback
# =============================================================================
class SelfEvaluatingAgentWithFeedbackConfig(FunctionBaseConfig, name="self_evaluating_agent_with_feedback"):
    """
    Configuration for Self-Evaluating Agent with Feedback Loop.

    This advanced version passes evaluation feedback to the agent on retry,
    allowing it to learn from previous attempts. Use this for better quality
    at the cost of slightly higher latency.
    """

    wrapped_agent: FunctionRef = Field(..., description="The underlying agent to wrap")
    evaluator_llm: LLMRef = Field(..., description="LLM to use for self-evaluation")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts", ge=0, le=5)
    min_confidence_threshold: float = Field(default=0.85,
                                            description="Minimum confidence to accept the tool sequence",
                                            ge=0.0,
                                            le=1.0)
    pass_feedback_to_agent: bool = Field(default=True,
                                         description="Whether to pass evaluation feedback to the agent on retry")
    feedback_template: str = Field(
        default="""PREVIOUS ATTEMPT FEEDBACK:

Your previous tool selection was evaluated and found to be insufficient.

EVALUATION:
{reasoning}

MISSING STEPS:
{missing_steps}

SUGGESTIONS:
{suggestions}

Please try again, addressing the issues identified above. Focus on:
1. Including all necessary information gathering steps
2. Ensuring proper order of operations
3. Adding verification steps where appropriate
""",
        description="Template for feedback passed to agent on retry",
    )
    evaluation_prompt_template: str = Field(
        default="""You are evaluating whether a sequence of tool calls is sufficient to answer a user's question.

USER QUESTION:
{question}

TOOL CALLS MADE:
{tool_calls}

Evaluate whether these tool calls would be sufficient to fully answer the user's question.
Consider:
1. Are all necessary information gathering steps included?
2. Are the tools called in the correct order?
3. Are there any missing steps or tools that should have been called?
4. Would the user's request be fully satisfied?

Respond with a JSON object:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "explanation of your evaluation",
    "missing_steps": ["list", "of", "missing", "steps"],
    "suggestions": "how to improve the tool sequence"
}}

JSON Response:""",
        description="Template for self-evaluation prompt",
    )
    verbose: bool = Field(default=True, description="Enable verbose logging")


@register_function(config_type=SelfEvaluatingAgentWithFeedbackConfig)
async def self_evaluating_agent_with_feedback_function(config: SelfEvaluatingAgentWithFeedbackConfig, builder: Builder):
    """
    Register the advanced self-evaluating agent with feedback loop.

    Args:
        config: Configuration for the agent
        builder: The builder object

    Yields:
        FunctionInfo: The function info for the agent
    """
    # Get the wrapped agent and evaluator LLM
    wrapped_agent = await builder.get_function(config.wrapped_agent)
    evaluator_llm = await builder.get_llm(config.evaluator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Get the tool intent buffer from runtime metadata (may be None if in different builder)
    intent_buffer = None
    if hasattr(builder, "runtime_metadata"):
        intent_buffer = builder.runtime_metadata.get("tool_intent_buffer")

    # Flag to use global registry as fallback
    use_global_registry = intent_buffer is None

    async def _self_evaluating_agent_with_feedback(question: str) -> str:
        """
        Execute the agent with self-evaluation, feedback, and retry.

        Args:
            question: The user's input question

        Returns:
            The final answer from the agent
        """
        attempt = 0
        best_result = None
        best_evaluation = None
        previous_feedback = None

        # Generate unique scenario ID for this question (for concurrent execution isolation)
        scenario_id = _get_scenario_id_for_question(question)

        # Log entry
        if config.verbose:
            logger.info("🚀 Starting self-evaluating agent with feedback")
            logger.info("   Max retries: %d", config.max_retries)
            logger.info("   Confidence threshold: %.2f", config.min_confidence_threshold)
            logger.info("   Pass feedback: %s", config.pass_feedback_to_agent)
            logger.info("   Scenario ID: %s", scenario_id)
            logger.info("   Question: %s", question[:100] + "..." if len(question) > 100 else question)

        # Debug buffer availability
        if intent_buffer:
            logger.info("✅ Intent buffer available (builder): %s", type(intent_buffer).__name__)
        elif use_global_registry:
            logger.info("✅ Using GLOBAL intent registry (scenario: %s)", scenario_id)
        else:
            logger.error("❌ NO INTENT BUFFER - Self-evaluation will not work!")

        while attempt <= config.max_retries:
            if config.verbose:
                logger.info("=" * 80)
                logger.info("🔄 SELF-EVALUATION ATTEMPT %d/%d", attempt + 1, config.max_retries + 1)
                if previous_feedback and config.pass_feedback_to_agent:
                    logger.info("   (With feedback from previous attempt)")
                logger.info("=" * 80)

            # Set the current scenario ID for this thread (so tool stubs record to correct scenario)
            set_current_scenario_id(scenario_id)

            # Clear the intent buffer/registry for this attempt
            if intent_buffer:
                intent_buffer.clear()
                logger.debug("🗑️  Cleared intent buffer for fresh attempt")
            else:
                # Use global registry with unique scenario ID
                clear_global_intents(scenario_id)
                logger.debug("🗑️  Cleared GLOBAL intent registry (scenario: %s)", scenario_id)

            # Construct the query (with feedback if this is a retry)
            if attempt > 0 and previous_feedback and config.pass_feedback_to_agent:
                # Append feedback to the original question
                query = f"{question}\n\n{previous_feedback}"
                if config.verbose:
                    logger.info("📝 Passing feedback to agent (%d chars):", len(previous_feedback))
                    logger.info("   Feedback preview:\n%s",
                                previous_feedback[:500] + "..." if len(previous_feedback) > 500 else previous_feedback)
            else:
                query = question
                if attempt > 0:
                    logger.info("🔁 Retry WITHOUT feedback (pass_feedback_to_agent=%s)", config.pass_feedback_to_agent)

            # Execute the wrapped agent
            try:
                logger.debug("⚙️  Executing wrapped agent...")
                result = await wrapped_agent.ainvoke(query)
                logger.debug("✓ Agent execution completed")
            except Exception:
                logger.exception("💥 Error executing wrapped agent on attempt %d", attempt + 1)
                attempt += 1
                continue

            # Get the tool calls from the buffer or global registry
            tool_calls = []
            if intent_buffer:
                tool_calls = intent_buffer.get_intents()
                logger.debug("Retrieved %d intents from builder buffer", len(tool_calls))
            else:
                # Fallback to global registry with unique scenario ID
                tool_calls = get_global_intents(scenario_id)
                logger.debug("Retrieved %d intents from GLOBAL registry (scenario: %s)", len(tool_calls), scenario_id)

            if config.verbose:
                source = "buffer" if intent_buffer else f"global registry ({scenario_id})"
                logger.info("📊 Captured %d tool calls from %s", len(tool_calls), source)
                for i, call in enumerate(tool_calls, 1):
                    tool_name = call.get("tool", "unknown")
                    param_count = len(call.get("parameters", {}))
                    logger.info("  %d. %s (params: %d)", i, tool_name, param_count)

            # Check if we have a valid way to track intents
            has_intent_tracking = intent_buffer is not None or use_global_registry

            # If this is the last attempt OR we can't track intents, accept the result
            if attempt >= config.max_retries or not has_intent_tracking:
                if config.verbose:
                    logger.info("🏁 Final attempt reached - accepting result")
                    logger.info("   Total attempts made: %d", attempt + 1)
                    logger.info("   Best confidence seen: %.2f",
                                best_evaluation.get("confidence", 0.0) if best_evaluation else 0.0)
                return result

            # Perform self-evaluation
            logger.debug("🔍 Starting self-evaluation...")
            evaluation_result = await _evaluate_tool_sequence(
                question=question,
                tool_calls=tool_calls,
                evaluator_llm=evaluator_llm,
                prompt_template=config.evaluation_prompt_template,
                verbose=config.verbose,
            )

            # Track the best result
            if best_evaluation is None or evaluation_result.get("confidence", 0) > best_evaluation.get("confidence", 0):
                best_result = result
                best_evaluation = evaluation_result
                if config.verbose:
                    logger.debug("📈 New best result (confidence: %.2f)", evaluation_result.get("confidence", 0))

            # Check if sufficient
            is_sufficient = evaluation_result.get("is_sufficient", False)
            confidence = evaluation_result.get("confidence", 0.0)

            if config.verbose:
                logger.info("-" * 80)
                logger.info("🔍 Self-Evaluation Result:")
                logger.info("  Sufficient: %s", is_sufficient)
                logger.info("  Confidence: %.2f (threshold: %.2f)", confidence, config.min_confidence_threshold)
                logger.info("  Reasoning: %s", evaluation_result.get("reasoning", "N/A"))
                if not is_sufficient:
                    missing = evaluation_result.get("missing_steps", [])
                    if missing:
                        logger.info("  Missing steps (%d): %s", len(missing), ", ".join(missing))
                    suggestions = evaluation_result.get("suggestions", "")
                    if suggestions:
                        logger.info("  Suggestions: %s", suggestions)
                logger.info("-" * 80)

            # Accept if sufficient and confident
            if is_sufficient and confidence >= config.min_confidence_threshold:
                if config.verbose:
                    logger.info("✅ Tool sequence ACCEPTED after %d attempt(s)", attempt + 1)
                    logger.info("   Final tool count: %d", len(tool_calls))
                return result

            # Prepare feedback for next attempt
            if config.pass_feedback_to_agent:
                missing_steps = evaluation_result.get("missing_steps", [])
                missing_steps_str = "\n".join(f"- {step}"
                                              for step in missing_steps) if missing_steps else "None identified"

                previous_feedback = config.feedback_template.format(
                    reasoning=evaluation_result.get("reasoning", "Insufficient tool sequence"),
                    missing_steps=missing_steps_str,
                    suggestions=evaluation_result.get("suggestions", "No specific suggestions"),
                )
                logger.debug("📋 Generated feedback (%d chars) for next attempt", len(previous_feedback))

            if config.verbose:
                logger.warning("❌ Tool sequence INSUFFICIENT - retrying with feedback...")
                logger.warning("   Reason: is_sufficient=%s, confidence=%.2f < threshold=%.2f",
                               is_sufficient,
                               confidence,
                               config.min_confidence_threshold)

            attempt += 1

        # All retries exhausted
        if config.verbose:
            logger.warning("⚠️  MAX RETRIES EXHAUSTED - returning best result")
            logger.warning("   Total attempts: %d", config.max_retries + 1)
            logger.warning("   Best confidence: %.2f",
                           best_evaluation.get("confidence", 0.0) if best_evaluation else 0.0)

        return best_result if best_result is not None else "No valid result obtained after retries."

    yield FunctionInfo.from_fn(
        _self_evaluating_agent_with_feedback,
        description="Advanced self-evaluating agent with feedback loop for improved retries",
    )


# =============================================================================
# LEGACY REGISTRATION: self_evaluating_agent (backward compatible)
# =============================================================================
@register_function(config_type=SelfEvaluatingAgentConfig)
async def self_evaluating_agent_function(config: SelfEvaluatingAgentConfig, builder: Builder):
    """
    Register the self-evaluating agent wrapper (legacy mode).

    This is a backward-compatible wrapper that uses the same implementation
    as the advanced version but with different defaults (no feedback by default).

    Args:
        config: Configuration for the self-evaluating agent
        builder: The builder object

    Yields:
        FunctionInfo: The function info for the self-evaluating agent
    """
    # Get the wrapped agent and evaluator LLM
    wrapped_agent = await builder.get_function(config.wrapped_agent)
    evaluator_llm = await builder.get_llm(config.evaluator_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

    # Get the tool intent buffer from runtime metadata (may be None if in different builder)
    intent_buffer = None
    if hasattr(builder, "runtime_metadata"):
        intent_buffer = builder.runtime_metadata.get("tool_intent_buffer")

    # Flag to use global registry as fallback
    use_global_registry = intent_buffer is None

    async def _self_evaluating_agent(question: str) -> str:
        """
        Execute the agent with self-evaluation and retry.

        Args:
            question: The user's input question

        Returns:
            The final answer from the agent
        """
        attempt = 0
        best_result = None
        best_evaluation = None
        previous_feedback = None

        # Generate unique scenario ID for this question (for concurrent execution isolation)
        scenario_id = _get_scenario_id_for_question(question)

        # Log entry
        if config.verbose:
            logger.info("🚀 Starting self-evaluating agent")
            logger.info("   Max retries: %d", config.max_retries)
            logger.info("   Confidence threshold: %.2f", config.min_confidence_threshold)
            logger.info("   Pass feedback: %s", config.pass_feedback_to_agent)
            logger.info("   Scenario ID: %s", scenario_id)
            logger.info("   Question: %s", question[:100] + "..." if len(question) > 100 else question)

        # Debug buffer availability
        if intent_buffer:
            logger.debug("✅ Intent buffer available (builder): %s", type(intent_buffer).__name__)
        elif use_global_registry:
            logger.debug("✅ Using GLOBAL intent registry (scenario: %s)", scenario_id)
        else:
            logger.error("❌ NO INTENT BUFFER - Self-evaluation will not work!")

        while attempt <= config.max_retries:
            if config.verbose:
                logger.info("=" * 80)
                logger.info("Attempt %d/%d", attempt + 1, config.max_retries + 1)
                if previous_feedback and config.pass_feedback_to_agent:
                    logger.info("   (With feedback from previous attempt)")
                logger.info("=" * 80)

            # Set the current scenario ID for this thread (so tool stubs record to correct scenario)
            set_current_scenario_id(scenario_id)

            # Clear the intent buffer/registry for this attempt
            if intent_buffer:
                intent_buffer.clear()
                logger.debug("Cleared intent buffer for fresh attempt")
            else:
                # Use global registry with unique scenario ID
                clear_global_intents(scenario_id)
                logger.debug("Cleared GLOBAL intent registry (scenario: %s)", scenario_id)

            # Construct the query (with feedback if this is a retry and feedback is enabled)
            if attempt > 0 and previous_feedback and config.pass_feedback_to_agent:
                # Append feedback to the original question
                query = f"{question}\n\n{previous_feedback}"
                if config.verbose:
                    logger.info("Passing feedback to agent (%d chars)", len(previous_feedback))
            else:
                query = question

            # Execute the wrapped agent
            try:
                result = await wrapped_agent.ainvoke(query)
            except Exception:
                logger.exception("Error executing wrapped agent on attempt %d", attempt + 1)
                attempt += 1
                continue

            # Get the tool calls from the buffer or global registry
            tool_calls = []
            if intent_buffer:
                tool_calls = intent_buffer.get_intents()
            else:
                # Fallback to global registry with unique scenario ID
                tool_calls = get_global_intents(scenario_id)

            if config.verbose:
                logger.info("Captured %d tool calls", len(tool_calls))
                for i, call in enumerate(tool_calls, 1):
                    logger.info("  %d. %s", i, call.get("tool", "unknown"))

            # Check if we have a valid way to track intents
            has_intent_tracking = intent_buffer is not None or use_global_registry

            # If this is the last attempt OR we can't track intents, accept the result
            if attempt >= config.max_retries or not has_intent_tracking:
                if config.verbose:
                    logger.info("Final attempt reached - accepting result")
                return result

            # Perform self-evaluation
            evaluation_result = await _evaluate_tool_sequence(
                question=question,
                tool_calls=tool_calls,
                evaluator_llm=evaluator_llm,
                prompt_template=config.evaluation_prompt_template,
                verbose=config.verbose,
            )

            # Track the best result
            if best_evaluation is None or evaluation_result.get("confidence", 0) > best_evaluation.get("confidence", 0):
                best_result = result
                best_evaluation = evaluation_result

            # Check if the tool sequence is sufficient
            is_sufficient = evaluation_result.get("is_sufficient", False)
            confidence = evaluation_result.get("confidence", 0.0)

            if config.verbose:
                logger.info("-" * 80)
                logger.info("Self-Evaluation Result:")
                logger.info("  Sufficient: %s", is_sufficient)
                logger.info("  Confidence: %.2f", confidence)
                logger.info("  Reasoning: %s", evaluation_result.get("reasoning", "N/A"))
                if not is_sufficient:
                    missing = evaluation_result.get("missing_steps", [])
                    if missing:
                        logger.info("  Missing steps: %s", ", ".join(missing))
                    suggestions = evaluation_result.get("suggestions", "")
                    if suggestions:
                        logger.info("  Suggestions: %s", suggestions)
                logger.info("-" * 80)

            # Accept if sufficient and confidence meets threshold
            if is_sufficient and confidence >= config.min_confidence_threshold:
                if config.verbose:
                    logger.info("✓ Tool sequence accepted (sufficient and confident)")
                return result

            # Prepare feedback for next attempt (if feedback is enabled)
            if config.pass_feedback_to_agent:
                missing_steps = evaluation_result.get("missing_steps", [])
                missing_steps_str = ("\n".join(f"- {step}"
                                               for step in missing_steps) if missing_steps else "None identified")
                previous_feedback = config.feedback_template.format(
                    reasoning=evaluation_result.get("reasoning", "Insufficient tool sequence"),
                    missing_steps=missing_steps_str,
                    suggestions=evaluation_result.get("suggestions", "No specific suggestions"),
                )

            # Otherwise, retry
            if config.verbose:
                logger.warning("✗ Tool sequence insufficient - retrying...")

            attempt += 1

        # All retries exhausted - return the best result
        if config.verbose:
            logger.warning(
                "Max retries exhausted - returning best result (confidence: %.2f)",
                best_evaluation.get("confidence", 0.0) if best_evaluation else 0.0,
            )

        return best_result if best_result is not None else "No valid result obtained after retries."

    yield FunctionInfo.from_fn(
        _self_evaluating_agent,
        description="Self-evaluating agent wrapper that validates tool call sequences and retries if insufficient",
    )


# =============================================================================
# SHARED UTILITY: Tool sequence evaluation
# =============================================================================
async def _evaluate_tool_sequence(
    question: str,
    tool_calls: list[dict[str, Any]],
    evaluator_llm: Any,
    prompt_template: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Evaluate whether a tool call sequence is sufficient."""
    # Format tool calls
    if not tool_calls:
        tool_calls_str = "No tool calls were made."
    else:
        tool_calls_formatted = []
        for i, call in enumerate(tool_calls, 1):
            tool_name = call.get("tool", "unknown")
            parameters = call.get("parameters", {})
            params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
            tool_calls_formatted.append(f"{i}. {tool_name}({params_str})")
        tool_calls_str = "\n".join(tool_calls_formatted)

    # Create the evaluation prompt
    prompt = prompt_template.format(question=question, tool_calls=tool_calls_str)

    if verbose:
        logger.debug("Evaluating tool sequence with %d calls", len(tool_calls))

    try:
        # Call the evaluator LLM
        response = await evaluator_llm.ainvoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        # Parse the JSON response
        # Find JSON in the response (it might have extra text)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
        else:
            # Fallback if no JSON found
            logger.warning("No JSON found in evaluation response, using defaults")
            result = {
                "is_sufficient": False,
                "confidence": 0.5,
                "reasoning": response_text[:500],
                "missing_steps": [],
                "suggestions": "Could not parse evaluation response",
            }

        return result

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse evaluation JSON: %s", e)
        return {
            "is_sufficient": False,
            "confidence": 0.5,
            "reasoning": "Failed to parse evaluation response",
            "missing_steps": [],
            "suggestions": "Retry with clearer tool sequence",
        }
    except Exception:
        logger.exception("Error during self-evaluation")
        return {
            "is_sufficient": False,
            "confidence": 0.0,
            "reasoning": "Evaluation error occurred",
            "missing_steps": [],
            "suggestions": "Check evaluation LLM configuration",
        }
