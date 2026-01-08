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
LLM Endpoint Validator for NAT Evaluation.

This module provides functionality to validate LLM endpoints before running evaluation
workflows. This helps catch deployment issues early (e.g., models not deployed after
training cancellation) and provides actionable error messages.

The validation uses NAT's WorkflowBuilder to instantiate LLMs in a framework-agnostic way,
then tests them with a minimal ainvoke() call. This approach works for all LLM types
(OpenAI, NIM, AWS Bedrock, vLLM, etc.) and respects NAT's native auth and config system.
"""

import logging
from typing import TYPE_CHECKING

from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.workflow_builder import WorkflowBuilder

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


def _is_404_error(exception: Exception) -> bool:
    """
    Detect if an exception represents a 404 (model not found) error.

    This handles various 404 error formats from different LLM providers:
    - OpenAI SDK: openai.NotFoundError
    - HTTP responses: 404 in error message
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

    # Check for 404 in error message
    if "404" in exception_str:
        return True

    # Check for common "not found" phrases
    if any(phrase in exception_str for phrase in ["not found", "does not exist", "not deployed"]):
        return True

    return False


def _get_llm_endpoint_info(llm_config) -> tuple[str | None, str | None]:
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

    Args:
        config: The NAT configuration object containing LLM definitions.

    Raises:
        RuntimeError: If any LLM endpoint has a 404 error (model not deployed).
    """
    if not config.llms:
        logger.debug("No LLMs defined in config, skipping validation")
        return

    failed_llms = []  # List of (llm_name, error_message) tuples for 404 errors
    validation_warnings = []  # List of (llm_name, warning_message) tuples for non-critical errors

    # Use WorkflowBuilder to instantiate and test LLMs
    async with WorkflowBuilder() as builder:
        for llm_name, llm_config in config.llms.items():
            try:
                logger.info("Validating LLM '%s' (type: %s)", llm_name, llm_config.type)

                # Add LLM to builder (handles all LLM types)
                await builder.add_llm(llm_name, llm_config)

                # Get LangChain-wrapped LLM instance
                llm = await builder.get_llm(llm_name, LLMFrameworkEnum.LANGCHAIN)

                # Test with minimal prompt - this will hit the endpoint
                try:
                    await llm.ainvoke("test")
                    logger.info("LLM '%s' validated successfully", llm_name)

                except Exception as invoke_error:
                    # Check if this is a 404 error (model not deployed)
                    if _is_404_error(invoke_error):
                        base_url, model_name = _get_llm_endpoint_info(llm_config)

                        error_msg = (
                            f"LLM '{llm_name}' validation failed: Model not found (404).\n"
                            f"This typically means:\n"
                            f"  1. The model has not been deployed yet\n"
                            f"  2. The model name is incorrect\n"
                            f"  3. A training job was canceled and the model was never deployed\n"
                            f"\nLLM Configuration:\n"
                            f"  Type: {llm_config.type}\n"
                            f"  Endpoint: {base_url or 'N/A'}\n"
                            f"  Model: {model_name or 'N/A'}\n"
                            f"\nACTION REQUIRED:\n"
                            f"  1. Verify the model is deployed: Check your deployment service\n"
                            f"  2. If using NeMo Customizer, ensure training completed successfully\n"
                            f"  3. Check model deployment status in your platform\n"
                            f"  4. Verify the model name matches the deployed model\n"
                            f"\nOriginal error: {invoke_error}"
                        )
                        logger.error(error_msg)
                        failed_llms.append((llm_name, error_msg))

                    else:
                        # Non-404 error - might be auth, rate limit, temporary issue, etc.
                        # Don't fail validation, but warn the user
                        warning_msg = (
                            f"Could not fully validate LLM '{llm_name}': {invoke_error}. "
                            f"This might be due to auth requirements, rate limits, or temporary issues. "
                            f"Evaluation will proceed, but may fail if the LLM is truly inaccessible."
                        )
                        logger.warning(warning_msg)
                        validation_warnings.append((llm_name, str(invoke_error)))

            except Exception as e:
                # Error during builder setup (before ainvoke)
                # This could be import errors, config errors, etc.
                warning_msg = f"Could not validate LLM '{llm_name}' due to setup error: {e}"
                logger.warning(warning_msg)
                validation_warnings.append((llm_name, str(e)))

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
        raise RuntimeError(
            f"LLM endpoint validation failed for {len(failed_llms)} LLM(s) with 404 errors:\n\n"
            f"{error_summary}\n\n"
            f"Evaluation cannot proceed with undeployed models. "
            f"Please resolve the deployment issues above before retrying."
        )

    logger.info("All LLM endpoints validated successfully")
