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

"""LLM endpoint validation utilities for evaluation."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nat.data_models.config import Config

logger = logging.getLogger(__name__)


async def validate_llm_endpoints(config: "Config") -> None:
    """
    Validate that all LLM endpoints in the config are accessible.

    This function checks OpenAI-compatible LLM endpoints to ensure they are
    reachable before running evaluation. This prevents cryptic 404 errors
    during evaluation when models are not deployed.

    Args:
        config: The NAT configuration object containing LLM definitions.

    Raises:
        RuntimeError: If any LLM endpoint validation fails.
    """
    if not config.llms:
        logger.debug("No LLMs defined in config, skipping validation")
        return

    failed_llms = []

    for llm_name, llm_config in config.llms.items():
        try:
            # Only validate OpenAI-compatible endpoints
            if llm_config.type not in ["openai", "nim"]:
                logger.debug("Skipping validation for LLM '%s' (type: %s)", llm_name, llm_config.type)
                continue

            base_url = getattr(llm_config, "base_url", None)
            model_name = getattr(llm_config, "model_name", None)

            if not base_url:
                logger.debug("LLM '%s' has no base_url, skipping validation", llm_name)
                continue

            logger.info("Validating LLM endpoint '%s': %s", llm_name, base_url)

            # Try to connect to the endpoint
            try:
                import openai

                # Get API key if available
                api_key = getattr(llm_config, "api_key", None) or "unused"

                # Create client and test connection
                client = openai.OpenAI(base_url=base_url, api_key=api_key)

                # Simple connectivity check - list models
                try:
                    client.models.list()  # Just check if endpoint is accessible
                    logger.info("LLM endpoint '%s' is accessible", llm_name)
                except openai.NotFoundError as nf_error:
                    # 404 means endpoint is reachable but model doesn't exist
                    error_msg = (
                        f"LLM endpoint '{llm_name}' is reachable but model '{model_name}' was not found (404). "
                        f"This typically means:\n"
                        f"  1. The model has not been deployed yet\n"
                        f"  2. The model name is incorrect\n"
                        f"  3. A training job was canceled and the model was never deployed\n"
                        f"\nEndpoint: {base_url}\n"
                        f"Model: {model_name}\n"
                        f"\nACTION REQUIRED:\n"
                        f"  1. Verify the model is deployed: Check your NIM deployment service\n"
                        f"  2. If using NeMo Customizer, ensure training completed successfully\n"
                        f"  3. Check model deployment status in your NeMo MS platform\n"
                        f"  4. Verify the model name matches the deployed model\n"
                        f"\nOriginal error: {nf_error}"
                    )
                    logger.error(error_msg)
                    failed_llms.append((llm_name, error_msg))
                    continue
                except (ConnectionError, OSError, TimeoutError) as conn_error:
                    # Connection errors are critical - re-raise to be caught by outer handler
                    raise
                except Exception as list_error:
                    # Other errors might be okay (auth, rate limit, etc.) or endpoint doesn't support /models
                    logger.warning(
                        "LLM endpoint '%s' validation inconclusive: %s. "
                        "Proceeding with evaluation (endpoint may still work).",
                        llm_name,
                        list_error,
                    )

            except ImportError:
                logger.warning(
                    "Cannot validate LLM '%s': openai package not installed. Install with: pip install openai", llm_name
                )
            except Exception as e:
                error_msg = (
                    f"Failed to connect to LLM endpoint '{llm_name}' at {base_url}: {e}\n"
                    f"\nACTION REQUIRED:\n"
                    f"  1. Check that the endpoint is running and accessible\n"
                    f"  2. Verify network connectivity to {base_url}\n"
                    f"  3. Ensure API credentials are correct\n"
                    f"  4. Check firewall and proxy settings\n"
                )
                logger.error(error_msg)
                failed_llms.append((llm_name, error_msg))

        except Exception as e:
            logger.warning("Error during validation of LLM '%s': %s", llm_name, e)

    # If any critical LLMs failed, raise an error
    if failed_llms:
        error_summary = "\n\n".join([f"LLM '{name}':\n{msg}" for name, msg in failed_llms])
        raise RuntimeError(
            f"LLM endpoint validation failed for {len(failed_llms)} LLM(s):\n\n"
            f"{error_summary}\n\n"
            f"Evaluation cannot proceed with inaccessible LLM endpoints. "
            f"Please resolve the issues above before retrying."
        )

    logger.info("All LLM endpoints validated successfully")
