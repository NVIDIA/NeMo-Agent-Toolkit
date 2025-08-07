# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Guardrails provider implementation."""

import logging
import tempfile
from typing import Any

from aiq.data_models.llm import LLMBaseConfig
from aiq.guardrails.interface import GuardrailsProvider

from .config import NemoGuardrailsConfig
from .prompts import PromptManager

logger = logging.getLogger(__name__)


def is_content_safe(content: str) -> bool:
    """
    Check if content is safe based on NeMo Guardrails approach.
    Adapted from NeMo Guardrails output_parsers.py
    """
    if not content:
        return True

    content_lower = content.lower().strip()

    # Check for explicit safety indicators
    if "safe" in content_lower:
        return True
    elif content_lower in ["unsafe", "yes"]:
        return False
    elif content_lower == "no":
        return True

    # Default to safe if no clear indicators
    return True


def _is_blocked_by_guardrails(response, activated_rails: list = None) -> tuple[bool, str]:
    """
    Determine if input was blocked by guardrails using NeMo Guardrails metadata.

    Returns:
        tuple: (is_blocked, guardrails_response)
    """
    # Method 1: Check rail metadata (most reliable)
    if activated_rails:
        for rail in activated_rails:
            # Check if rail stopped execution
            if hasattr(rail, 'stop') and getattr(rail, 'stop', False):
                logger.debug("Rail '%s' stopped execution", getattr(rail, 'name', 'unknown'))

                # Get the guardrails response
                content = ""
                if isinstance(response, dict):
                    content = response.get("content", "")
                elif hasattr(response, 'content'):
                    content = getattr(response, 'content', '')
                elif isinstance(response, str):
                    content = response

                return True, content

            # Check for 'refuse to respond' in decisions
            if hasattr(rail, 'decisions'):
                decisions = getattr(rail, 'decisions', [])
                if isinstance(decisions, list) and 'refuse to respond' in decisions:
                    logger.debug("Rail '%s' refused to respond", getattr(rail, 'name', 'unknown'))

                    # Get the guardrails response
                    content = ""
                    if isinstance(response, dict):
                        content = response.get("content", "")
                    elif hasattr(response, 'content'):
                        content = getattr(response, 'content', '')
                    elif isinstance(response, str):
                        content = response

                    return True, content

    # Method 2: Content-based analysis (fallback)
    content = ""
    if isinstance(response, dict):
        content = response.get("content", "")
    elif hasattr(response, 'content'):
        content = getattr(response, 'content', '')
    elif isinstance(response, str):
        content = response

    if content:
        # Check for common blocking phrases
        blocking_phrases = [
            "i'm sorry, i can't",
            "i cannot",
            "i'm not able to",
            "i won't be able to",
            "i can't provide",
            "i'm unable to",
            "sorry, but i can't"
        ]

        content_lower = content.lower()
        for phrase in blocking_phrases:
            if phrase in content_lower:
                logger.debug("Content-based blocking detected: '%s'", phrase)
                return True, content

        # Use content safety check
        if not is_content_safe(content):
            logger.debug("Content marked as unsafe by safety check")
            return True, content

    return False, ""


class NemoGuardrailsProvider(GuardrailsProvider):
    """NeMo Guardrails provider implementation."""

    def __init__(self, config: NemoGuardrailsConfig, llm_config: LLMBaseConfig | None = None):
        super().__init__(config, llm_config)
        if not isinstance(config, NemoGuardrailsConfig):
            raise ValueError("Config must be NemoGuardrailsConfig")
        self._rails = None
        self.prompt_manager = PromptManager()

    async def initialize(self) -> None:
        """Initialize NeMo Guardrails."""
        try:
            from nemoguardrails import LLMRails
            from nemoguardrails import RailsConfig

            config = self.config

            # Handle optional config_path
            if config.config_path:
                # Load configuration from specified path
                rails_config = RailsConfig.from_path(config.config_path)
                logger.debug("Loaded NeMo Guardrails configuration from %s", config.config_path)
            else:
                # Generate configuration with default prompts for known rails
                yaml_content, colang_content = self._generate_config_with_defaults(config)

                # Override LLM if specified in guardrails config
                if self.llm_config and config.llm_name:
                    # Convert AIQ LLM to NeMo Guardrails format
                    nemo_model_config = self._convert_llm_config(self.llm_config)

                    # Parse the existing YAML to modify it
                    import yaml
                    yaml_dict = yaml.safe_load(yaml_content)
                    yaml_dict['models'] = [nemo_model_config]

                    # Convert back to YAML string
                    yaml_content_with_model = yaml.dump(yaml_dict, default_flow_style=False, allow_unicode=True)

                    rails_config = RailsConfig.from_content(colang_content=colang_content,
                                                            yaml_content=yaml_content_with_model)
                    logger.debug("Using AIQ LLM '%s' for guardrails: %s", config.llm_name, nemo_model_config)
                else:
                    rails_config = RailsConfig.from_content(colang_content=colang_content, yaml_content=yaml_content)
                    logger.debug("Generated NeMo Guardrails configuration with default prompts")
                    if not config.config_path:
                        # If no config_path and no LLM specified, we need at least a default model
                        logger.warning(
                            "No config_path or llm_name specified for guardrails - guardrails may not function properly"
                        )

            self._rails = LLMRails(rails_config, verbose=config.verbose)
            self._mark_initialized()
            logger.info("Successfully initialized NeMo Guardrails")

        except ImportError:
            logger.error("NeMo Guardrails not installed. Install with: pip install nemoguardrails")
            raise
        except Exception as e:
            logger.error("Failed to initialize NeMo Guardrails: %s", e)
            if getattr(self.config, 'fallback_on_error', False):
                logger.warning("Continuing without guardrails due to initialization error")
                self._rails = None
                self._mark_initialized()
            else:
                raise

    async def apply_input_guardrails(self, input_data: Any) -> tuple[Any, bool]:
        """Apply NeMo Guardrails to input."""
        input_rails_enabled = getattr(self.config, 'input_rails_enabled', True)
        if not getattr(self.config, 'enabled', True) or not input_rails_enabled or not self._rails:
            return input_data, True

        try:
            # Convert input to chat format if needed
            if hasattr(input_data, 'messages'):
                # AIQChatRequest format
                messages = input_data.messages
                last_message = messages[-1].content if messages else ""
            else:
                # String format
                last_message = str(input_data)

            # Enable logging to track activated rails
            options = {"log": {"activated_rails": True, "llm_calls": True, "internal_events": True}}

            # Apply input guardrails
            response = await self._rails.generate_async(messages=[{
                "role": "user", "content": last_message
            }],
                                                        options=options)

            # Extract activated rails for blocking detection
            activated_rails = []
            try:
                if hasattr(response, 'log') and hasattr(response.log,
                                                        'activated_rails') and response.log.activated_rails:
                    activated_rails = response.log.activated_rails
                    logger.debug("%d guardrails were activated", len(activated_rails))
                    for rail in activated_rails:
                        rail_info = "  - %s: %s" % (getattr(rail, 'type', 'unknown'), getattr(rail, 'name', 'unknown'))
                        if hasattr(rail, 'duration'):
                            rail_info += " (Duration: %.2fs)" % rail.duration
                        if hasattr(rail, 'decisions'):
                            rail_info += " (Decisions: %s)" % rail.decisions
                        if hasattr(rail, 'stop'):
                            rail_info += " (Stopped: %s)" % rail.stop
                        logger.debug(rail_info)
                else:
                    logger.debug("No input guardrails were activated")
            except Exception as e:
                logger.debug("Could not extract guardrails logging info: %s", e)

            # Use proper blocking detection
            is_blocked, guardrails_response = _is_blocked_by_guardrails(response, activated_rails)

            if is_blocked:
                logger.warning("Input blocked by guardrails: '%s'", guardrails_response)
                # Return the actual guardrails response, not the original input
                return guardrails_response, False

            return input_data, True

        except Exception as e:
            logger.error("Error applying input guardrails: %s", e)
            if getattr(self.config, 'fallback_on_error', False):
                logger.warning("Continuing without input guardrails due to error")
                return input_data, True
            else:
                raise

    async def apply_output_guardrails(self, output_data: Any, input_data: Any = None) -> Any:
        """Apply NeMo Guardrails to output using correct format."""
        # Check each condition separately
        enabled = getattr(self.config, 'enabled', True)
        output_rails_enabled = getattr(self.config, 'output_rails_enabled', True)
        has_rails = bool(self._rails)

        logger.debug("Output guardrails check: enabled=%s, output_rails_enabled=%s, has_rails=%s",
                     enabled,
                     output_rails_enabled,
                     has_rails)
        logger.debug("Config type: %s, Output data type: %s", type(self.config), type(output_data))

        if not enabled or not output_rails_enabled or not has_rails:
            logger.debug("Output guardrails skipped")
            return output_data

        try:
            # Extract content from output
            if hasattr(output_data, 'choices') and output_data.choices:
                # AIQChatResponse format
                content = output_data.choices[0].message.content
            else:
                # String format
                content = str(output_data)

            # Extract original user message from input_data for context
            user_message = ""
            if input_data:
                if hasattr(input_data, 'messages'):
                    # AIQChatRequest format
                    messages = input_data.messages
                    user_message = messages[-1].content if messages else ""
                else:
                    # String format
                    user_message = str(input_data)

            # Enable logging to track activated rails
            options = {"log": {"activated_rails": True, "llm_calls": True, "internal_events": True}}

            # Use NeMo Guardrails output checking format
            messages = [{
                "role": "context", "content": {
                    "llm_output": content
                }
            }, {
                "role": "user", "content": user_message
            }]

            response = await self._rails.generate_async(messages=messages, options=options)

            # Extract activated rails for blocking detection
            activated_rails = []
            try:
                if hasattr(response, 'log') and hasattr(response.log,
                                                        'activated_rails') and response.log.activated_rails:
                    activated_rails = response.log.activated_rails

                    # Filter to only show output-type rails for accurate logging
                    output_rails = [rail for rail in activated_rails if getattr(rail, 'type', '') == 'output']
                    total_rails = len(activated_rails)

                    if output_rails:
                        logger.debug("%d output guardrails were activated (out of %d total rails)",
                                     len(output_rails),
                                     total_rails)
                        for rail in output_rails:
                            rail_info = "  - %s: %s" % (getattr(rail, 'type', 'unknown'),
                                                        getattr(rail, 'name', 'unknown'))
                            if hasattr(rail, 'duration'):
                                rail_info += " (Duration: %.2fs)" % rail.duration
                            if hasattr(rail, 'decisions'):
                                rail_info += " (Decisions: %s)" % rail.decisions
                            if hasattr(rail, 'stop'):
                                rail_info += " (Stopped: %s)" % rail.stop
                            logger.debug(rail_info)
                    else:
                        logger.debug("No output guardrails were activated (but %d total rails ran)", total_rails)

                    # For debugging, show all rails that ran during this session
                    if total_rails > 0:
                        logger.debug("All %d rails that ran during this session:", total_rails)
                        for rail in activated_rails:
                            rail_info = "  - %s: %s" % (getattr(rail, 'type', 'unknown'),
                                                        getattr(rail, 'name', 'unknown'))
                            if hasattr(rail, 'duration'):
                                rail_info += " (Duration: %.2fs)" % rail.duration
                            logger.debug(rail_info)
                else:
                    logger.debug("No guardrails information available")
            except Exception as e:
                logger.debug("Could not extract output guardrails logging info: %s", e)

            # Use proper blocking detection (same as input guardrails)
            is_blocked, guardrails_response = _is_blocked_by_guardrails(response, activated_rails)

            if is_blocked:
                logger.warning("Output blocked by guardrails: '%s'", guardrails_response)
                # Return the blocked response in the same format as original output
                if hasattr(output_data, 'choices') and output_data.choices:
                    output_data.choices[0].message.content = guardrails_response
                else:
                    output_data = guardrails_response
                return output_data

            return output_data

        except Exception as e:
            logger.error("Error applying output guardrails: %s", e)
            if getattr(self.config, 'fallback_on_error', False):
                logger.warning("Continuing without output guardrails due to error")
                return output_data
            else:
                raise

    def create_fallback_response(self, input_data: Any) -> Any:
        """Create a fallback response when guardrails block execution."""
        fallback_message = getattr(self.config, 'fallback_response', "I cannot provide a response to that request.")

        # Create response in the same format as expected output
        if hasattr(input_data, 'messages'):
            # AIQChatRequest input -> AIQChatResponse output
            from aiq.data_models.api_server import AIQChatResponse
            return AIQChatResponse.from_string(fallback_message)
        else:
            # String input -> String output
            return fallback_message

    def _generate_config_with_defaults(self, config: NemoGuardrailsConfig):
        """Generate NeMo Guardrails configuration with default prompts for known rails."""
        # Extract flows from the rails configuration
        input_flows = self._get_flows_from_config(config, 'input')
        output_flows = self._get_flows_from_config(config, 'output')

        # Extract custom prompts from the rails configuration
        custom_prompts = {}
        if config.rails and 'prompts' in config.rails:
            custom_prompts = config.rails['prompts']
            logger.debug("Using custom prompts for: %s", list(custom_prompts.keys()))

        # Use prompt manager to generate configuration
        rails_config = self.prompt_manager.generate_nemo_guardrails_config(input_flows=input_flows,
                                                                           output_flows=output_flows,
                                                                           custom_prompts=custom_prompts)

        # Convert to yaml string for NeMo Guardrails with proper formatting
        import yaml

        # Custom representer for multi-line strings to ensure proper YAML formatting
        def str_presenter(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, str_presenter)

        yaml_content = yaml.dump(rails_config['yaml_content'], default_flow_style=False, allow_unicode=True)
        colang_content = rails_config['colang_content']

        logger.debug("Generated configuration with flows - Input: %s, Output: %s", input_flows, output_flows)

        # Debug logging to see what's being generated
        logger.debug("Generated YAML content:\n%s", yaml_content)
        if colang_content:
            logger.debug("Generated Colang content:\n%s", colang_content)
        else:
            logger.debug("No Colang content generated")

        # Check if prompts are in the yaml content
        if 'prompts' in rails_config['yaml_content']:
            logger.debug("Prompts found in yaml_content: %s", rails_config['yaml_content']['prompts'])
        else:
            logger.warning("No prompts found in yaml_content")

        return yaml_content, colang_content

    def _get_flows_from_config(self, config: NemoGuardrailsConfig, flow_type: str):
        """Extract flows from the rails configuration."""
        if not config.rails:
            return []

        flows_config = config.rails.get(flow_type, {})
        if isinstance(flows_config, dict):
            return flows_config.get('flows', [])
        elif isinstance(flows_config, list):
            return flows_config
        else:
            return []

    def _convert_llm_config(self, llm_config: LLMBaseConfig) -> dict[str, Any]:
        """Convert AIQ LLM config to NeMo Guardrails model config."""

        # Get the AIQ LLM type (e.g., "openai", "nim", "aws_bedrock")
        aiq_llm_type = llm_config.static_type()

        # Base model configuration
        nemo_config = {"type": "main", "parameters": {}}

        # Map AIQ LLM types to NeMo Guardrails engines
        if aiq_llm_type == "openai":
            nemo_config["engine"] = "openai"
            nemo_config["model"] = getattr(llm_config, 'model_name', 'gpt-3.5-turbo')

            # Add OpenAI-specific parameters
            api_key = getattr(llm_config, 'api_key', None)
            if api_key:
                nemo_config["parameters"]["api_key"] = api_key
            base_url = getattr(llm_config, 'base_url', None)
            if base_url:
                nemo_config["parameters"]["base_url"] = base_url
            temperature = getattr(llm_config, 'temperature', None)
            if temperature is not None:
                nemo_config["parameters"]["temperature"] = temperature
            top_p = getattr(llm_config, 'top_p', None)
            if top_p is not None:
                nemo_config["parameters"]["top_p"] = top_p
            seed = getattr(llm_config, 'seed', None)
            if seed:
                nemo_config["parameters"]["seed"] = seed

        elif aiq_llm_type == "nim":
            nemo_config["engine"] = "nim"
            nemo_config["model"] = getattr(llm_config, 'model_name', 'meta/llama3-8b-instruct')

            # Add NIM-specific parameters
            api_key = getattr(llm_config, 'api_key', None)
            if api_key:
                nemo_config["parameters"]["api_key"] = api_key
            base_url = getattr(llm_config, 'base_url', None)
            if base_url:
                nemo_config["parameters"]["base_url"] = base_url
            temperature = getattr(llm_config, 'temperature', None)
            if temperature is not None:
                nemo_config["parameters"]["temperature"] = temperature
            top_p = getattr(llm_config, 'top_p', None)
            if top_p is not None:
                nemo_config["parameters"]["top_p"] = top_p
            max_tokens = getattr(llm_config, 'max_tokens', None)
            if max_tokens:
                nemo_config["parameters"]["max_tokens"] = max_tokens

        elif aiq_llm_type == "aws_bedrock":
            nemo_config["engine"] = "bedrock"
            nemo_config["model"] = getattr(llm_config, 'model_name', 'anthropic.claude-v2')

            # Add AWS Bedrock-specific parameters
            region_name = getattr(llm_config, 'region_name', None)
            if region_name:
                nemo_config["parameters"]["region_name"] = region_name
            temperature = getattr(llm_config, 'temperature', None)
            if temperature is not None:
                nemo_config["parameters"]["temperature"] = temperature
            max_tokens = getattr(llm_config, 'max_tokens', None)
            if max_tokens:
                nemo_config["parameters"]["max_tokens"] = max_tokens

        else:
            # Fallback for unknown LLM types
            logger.warning("Unknown AIQ LLM type '%s', using generic mapping", aiq_llm_type)
            nemo_config["engine"] = "openai"  # Default to OpenAI-compatible
            nemo_config["model"] = getattr(llm_config, 'model_name', 'gpt-3.5-turbo')

            # Try to extract common parameters
            temperature = getattr(llm_config, 'temperature', None)
            if temperature is not None:
                nemo_config["parameters"]["temperature"] = temperature

        logger.debug("Converted AIQ LLM config to NeMo Guardrails: %s", nemo_config)
        return nemo_config
