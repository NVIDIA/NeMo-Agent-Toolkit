# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
LLM Endpoint Validator for NeMo Agent Toolkit evaluation.

This module provides functionality to validate LLM endpoints before running evaluation
workflows. This helps catch deployment issues early (e.g., models not deployed after
training cancellation) and provides actionable error messages.

The validation uses the NeMo Agent Toolkit `WorkflowBuilder` to instantiate LLMs in a framework-agnostic way,
then tests them with a minimal `ainvoke()` call. This approach works for all LLM types
(OpenAI, NIM, AWS Bedrock, vLLM, etc.) and respects the auth and config system.

Note: Validation invokes actual LLM endpoints with minimal test prompts. This may incur
small API costs for cloud-hosted models.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder
from nat.data_models.llm import LLMBaseConfig

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)

# Constants
VALIDATION_TIMEOUT_SECONDS = 30  # Timeout for each LLM validation
MAX_ERROR_MESSAGE_LENGTH = 500  # Truncate long error messages
CONCURRENT_VALIDATION_BATCH_SIZE = 5  # Max LLMs to validate in parallel
VALIDATION_PROMPT = "test"  # Minimal prompt for endpoint validation


def _is_404_error(exception: Exception) -> bool:
    """
    Detect if an exception represents a 404 (model not found) error.

    This handles various 404 error formats from different LLM providers:
    - OpenAI SDK: openai.NotFoundError
    - HTTP responses: HTTP 404 or status code 404
    - LangChain wrappers: Various wrapped 404s

    Args:
        exception: The exception to check.

    Returns:
        True if this is a 404 error, False otherwise.
    """
    exception_str = str(exception).lower()
    exception_type = type(exception).__name__

    # Check for NotFoundError type (OpenAI SDK)
    if "notfounderror" in exception_type.lower():
        return True

    # Check for HTTP 404 specifically (not just "404" which could appear in other contexts)
    if any(pattern in exception_str for pattern in ["http 404", "status code 404", "status_code=404"]):
        return True

    # Check for model-specific not found errors
    if "model" in exception_str and any(phrase in exception_str
                                        for phrase in ["not found", "does not exist", "not deployed", "not available"]):
        return True

    return False


def _get_llm_endpoint_info(llm_config: LLMBaseConfig) -> tuple[str | None, str | None]:
    """
    Extract endpoint and model information from an LLM config.

    Args:
        llm_config: The LLM configuration object.

    Returns:
        Tuple of (base_url, model_name), either may be None.
    """
    base_url = getattr(llm_config, "base_url", None)

    # Try multiple attributes for model name
    model_name = getattr(llm_config, "model_name", None)
    if model_name is None:
        model_name = getattr(llm_config, "model", None)

    return base_url, model_name


def _truncate_error_message(message: str, max_length: int = MAX_ERROR_MESSAGE_LENGTH) -> str:
    """
    Truncate error messages to prevent memory issues with large stack traces.

    Keeps both the start and end of the message to preserve context from both
    the error description (start) and the stack trace (end).

    Args:
        message: The error message to truncate.
        max_length: Maximum length to keep.

    Returns:
        Truncated message with ellipsis if needed.
    """
    if len(message) <= max_length:
        return message

    # Keep first and last portions to preserve both error description and stack trace
    separator = " ... (truncated) ... "

    # Guard for very small max_length values
    if max_length <= len(separator) + 2:
        return message[:max_length]

    keep_length = (max_length - len(separator)) // 2
    return f"{message[:keep_length]}{separator}{message[-keep_length:]}"


async def _validate_single_llm(builder: WorkflowBuilder, llm_name: str,
                               llm_config: LLMBaseConfig) -> tuple[str | None, str | None]:
    """
    Validate a single LLM endpoint.

    Args:
        builder: The WorkflowBuilder instance.
        llm_name: Name of the LLM to validate.
        llm_config: Configuration for the LLM.

    Returns:
        Tuple of (error_type, error_message):
        - error_type: "404" for model not found, "warning" for non-critical, None for success
        - error_message: Description of the error, or None if successful
    """
    try:
        logger.info("Validating LLM '%s' (type: %s)", llm_name, llm_config.type)
        start_time = time.time()

        # Add LLM to builder (handles all LLM types)
        await builder.add_llm(llm_name, llm_config)

        # Try all frameworks to find one that works with this LLM
        llm = None
        for framework in LLMFrameworkEnum:
            try:
                llm = await builder.get_llm(llm_name, framework)
                logger.debug("LLM '%s' successfully loaded with framework '%s'", llm_name, framework.value)
                break  # Found a working framework
            except Exception as e:
                logger.debug("LLM '%s' failed with framework '%s': %s", llm_name, framework.value, e)
                continue  # Try next framework

        if llm is None:
            # Log all attempted frameworks for debugging
            attempted = [f.value for f in LLMFrameworkEnum]
            error_msg = (f"Could not instantiate LLM '{llm_name}' with any known framework. "
                         f"Attempted: {', '.join(attempted)}. "
                         f"If this LLM uses a custom framework, this warning can be safely ignored. "
                         f"Otherwise, verify the LLM type '{llm_config.type}' is supported and configured correctly.")
            logger.warning("LLM '%s' - Framework instantiation failed: %s", llm_name, error_msg)
            return ("warning", error_msg)

        # Test with minimal prompt - this will hit the endpoint
        await asyncio.wait_for(llm.ainvoke(VALIDATION_PROMPT), timeout=VALIDATION_TIMEOUT_SECONDS)

        duration = time.time() - start_time
        logger.info("LLM '%s' validated successfully in %.2fs", llm_name, duration)
        return (None, None)

    except TimeoutError:
        error_msg = f"Validation timed out after {VALIDATION_TIMEOUT_SECONDS}s"
        logger.warning("LLM '%s' validation timed out", llm_name)
        return ("warning", _truncate_error_message(error_msg))

    except (KeyboardInterrupt, SystemExit):
        # Don't catch system-level interrupts
        raise

    except Exception as invoke_error:
        # Check if this is a 404 error (model not deployed)
        if _is_404_error(invoke_error):
            base_url, model_name = _get_llm_endpoint_info(llm_config)

            error_msg = (f"LLM '{llm_name}' validation failed: Model not found (404).\n"
                         f"\nThis typically means:\n"
                         f"  1. The model has not been deployed yet\n"
                         f"  2. The model name is incorrect\n"
                         f"  3. A training job was canceled and the model was never deployed\n"
                         f"\nLLM Configuration:\n"
                         f"  Type: {str(llm_config.type)}\n"
                         f"  Endpoint: {base_url or 'N/A'}\n"
                         f"  Model: {model_name or 'N/A'}\n"
                         f"\nACTION REQUIRED:\n"
                         f"  1. Verify the model is deployed (check your deployment service)\n"
                         f"  2. If using NeMo Customizer, ensure training completed successfully\n"
                         f"  3. Check model deployment status in your platform\n"
                         f"  4. Verify the model name matches the deployed model\n"
                         f"\nOriginal error: {_truncate_error_message(str(invoke_error))}")
            logger.exception(error_msg)
            return ("404", error_msg)

        else:
            # Non-404 error - might be auth, rate limit, temporary issue, etc.
            error_msg = (f"Could not fully validate LLM '{llm_name}': {_truncate_error_message(str(invoke_error))}. "
                         f"This might be due to auth requirements, rate limits, or temporary issues. "
                         f"Evaluation will proceed, but may fail if the LLM is truly inaccessible.")
            logger.exception(error_msg)
            return ("warning", _truncate_error_message(error_msg))


async def validate_llm_endpoints(config: "Config") -> None:
    """
    Validate that all LLM endpoints in the config are accessible.

    This function uses NAT's WorkflowBuilder to instantiate each configured LLM
    and tests it with a minimal ainvoke() call. This approach is framework-agnostic
    and works for all LLM types (OpenAI, NIM, AWS Bedrock, vLLM, etc.).

    The validation distinguishes between critical errors (404s indicating model not
    deployed) and non-critical errors (auth issues, rate limits, etc.):
    - 404 errors: Fail fast with detailed troubleshooting guidance
    - Other errors: Log warning but continue (to avoid false positives)

    LLMs are validated in parallel batches to improve performance while respecting
    rate limits. Each validation has a timeout to prevent hanging.

    Note: This function invokes actual LLM endpoints, which may incur small API costs.

    Args:
        config: The NAT configuration object containing LLM definitions.

    Raises:
        RuntimeError: If any LLM endpoint has a 404 error (model not deployed).
        ValueError: If config.llms is not properly structured.
    """

    # Validate config structure
    if not hasattr(config, "llms"):
        raise ValueError("Config does not have 'llms' attribute. Cannot validate LLM endpoints.")

    if not isinstance(config.llms, dict):
        raise ValueError(
            f"Config.llms must be a dict, got {type(config.llms).__name__}. Cannot validate LLM endpoints.")

    if not config.llms:
        logger.info("No LLMs configured - skipping endpoint validation")
        return

    failed_llms = []  # List of (llm_name, error_message) tuples for 404 errors
    validation_warnings = []  # List of (llm_name, warning_message) tuples for non-critical errors

    # Use WorkflowBuilder to instantiate and test LLMs
    async with WorkflowBuilder() as builder:
        # Get list of LLMs to validate
        llm_items = list(config.llms.items())

        # Validate in batches to respect rate limits
        for batch_start in range(0, len(llm_items), CONCURRENT_VALIDATION_BATCH_SIZE):
            batch = llm_items[batch_start:batch_start + CONCURRENT_VALIDATION_BATCH_SIZE]

            # Validate batch in parallel
            validation_tasks = [_validate_single_llm(builder, llm_name, llm_config) for llm_name, llm_config in batch]

            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results - zip with batch to maintain llm_name association
            for (llm_name, _llm_config), result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    # Re-raise system interrupts if they somehow got through
                    if isinstance(result, KeyboardInterrupt | SystemExit):
                        raise result

                    # Unexpected exception during validation
                    logger.warning("Unexpected error during validation: %s", _truncate_error_message(str(result)))
                    validation_warnings.append((llm_name, _truncate_error_message(str(result))))
                else:
                    # Normal result: (error_type, error_message)
                    error_type, error_message = result

                    if error_type == "404":
                        failed_llms.append((llm_name, error_message))
                    elif error_type == "warning":
                        validation_warnings.append((llm_name, error_message))
                    # If error_type is None, validation succeeded (no action needed)

    # Calculate validation metrics
    total_llms = len(llm_items)
    succeeded_count = total_llms - len(failed_llms) - len(validation_warnings)

    # Report non-critical warnings
    if validation_warnings:
        warning_summary = "\n".join([f"  - {name}: {msg}" for name, msg in validation_warnings])
        logger.warning(
            "LLM validation completed with %d warning(s):\n%s\nThese LLMs may still work during evaluation.",
            len(validation_warnings),
            warning_summary,
        )

    # If any LLMs have 404 errors, fail validation
    if failed_llms:
        error_summary = "\n\n".join([f"LLM '{name}':\n{msg}" for name, msg in failed_llms])

        # Log metrics before raising error
        logger.error(
            "Validation summary: %d total, %d succeeded, %d warned, %d failed (404)",
            total_llms,
            succeeded_count,
            len(validation_warnings),
            len(failed_llms),
        )

        raise RuntimeError(f"LLM endpoint validation failed for {len(failed_llms)} LLM(s) with 404 errors:\n\n"
                           f"{error_summary}\n\n"
                           f"Evaluation cannot proceed with undeployed models. "
                           f"Please resolve the deployment issues above before retrying.")

    # Log success metrics
    logger.info(
        "All LLM endpoints validated successfully - %d total, %d succeeded, %d warned",
        total_llms,
        succeeded_count,
        len(validation_warnings),
    )
