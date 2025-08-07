# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiq.guardrails.providers.nemo.config import NemoGuardrailsConfig


class TestNemoGuardrailsConfig:
    """Test the NemoGuardrailsConfig configuration class."""

    def test_config_creation_minimal(self):
        """Test creating config with minimal/default fields."""
        config = NemoGuardrailsConfig()

        # Check default values
        assert config.enabled is True
        assert config.input_rails_enabled is True
        assert config.output_rails_enabled is True
        assert config.config_path is None
        assert config.llm_name is None
        assert config.fallback_response == "I cannot provide a response to that request."
        assert config.fallback_on_error is True
        assert config.verbose is False
        assert config.max_retries == 3
        assert config.timeout_seconds == 30.0
        assert config.rails is None

    def test_config_creation_all_fields(self):
        """Test creating config with all fields specified."""
        custom_rails = {"input": {"flows": ["self check input"]}}

        config = NemoGuardrailsConfig(enabled=False,
                                      input_rails_enabled=False,
                                      output_rails_enabled=False,
                                      config_path="/path/to/config",
                                      llm_name="custom_llm",
                                      fallback_response="Custom fallback message",
                                      fallback_on_error=False,
                                      verbose=True,
                                      max_retries=5,
                                      timeout_seconds=60.0,
                                      rails=custom_rails)

        assert config.enabled is False
        assert config.input_rails_enabled is False
        assert config.output_rails_enabled is False
        assert config.config_path == "/path/to/config"
        assert config.llm_name == "custom_llm"
        assert config.fallback_response == "Custom fallback message"
        assert config.fallback_on_error is False
        assert config.verbose is True
        assert config.max_retries == 5
        assert config.timeout_seconds == 60.0
        assert config.rails == custom_rails

    def test_config_invalid_types(self):
        """Test that invalid types raise validation error."""
        # Invalid boolean type
        with pytest.raises(ValidationError):
            NemoGuardrailsConfig(enabled="invalid")  # Should be boolean

        # Invalid int type
        with pytest.raises(ValidationError):
            NemoGuardrailsConfig(max_retries="invalid")  # Should be int

        # Invalid float type
        with pytest.raises(ValidationError):
            NemoGuardrailsConfig(timeout_seconds="invalid")  # Should be float

    def test_config_serialization(self):
        """Test config serialization to dict."""
        config = NemoGuardrailsConfig(llm_name="test_llm", fallback_on_error=False)

        config_dict = config.model_dump()

        assert config_dict["llm_name"] == "test_llm"
        assert config_dict["fallback_on_error"] is False
        assert "enabled" in config_dict
        assert "input_rails_enabled" in config_dict

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "llm_name": "test_llm",
            "config_path": "/test/path",
            "fallback_on_error": False,
            "fallback_response": "Custom message"
        }

        config = NemoGuardrailsConfig(**config_dict)

        assert config.llm_name == "test_llm"
        assert config.config_path == "/test/path"
        assert config.fallback_on_error is False
        assert config.fallback_response == "Custom message"

    def test_config_type_name(self):
        """Test that config has correct type name."""
        config = NemoGuardrailsConfig()

        # The config should have the correct type name from the decorator
        assert hasattr(config, 'static_type')
        assert config.static_type() == "nemo_guardrails"

    def test_config_inheritance(self):
        """Test that config properly inherits from GuardrailsBaseConfig."""
        config = NemoGuardrailsConfig()

        # Should inherit from GuardrailsBaseConfig
        from aiq.data_models.guardrails import GuardrailsBaseConfig
        assert isinstance(config, GuardrailsBaseConfig)

    def test_config_optional_fields_none(self):
        """Test that optional fields can be set to None."""
        config = NemoGuardrailsConfig(config_path=None,
                                      llm_name=None,
                                      fallback_response=None,
                                      timeout_seconds=None,
                                      rails=None)

        assert config.config_path is None
        assert config.llm_name is None
        assert config.fallback_response is None
        assert config.timeout_seconds is None
        assert config.rails is None

    def test_config_rails_empty(self):
        """Test that rails defaults to None."""
        config = NemoGuardrailsConfig()

        assert config.rails is None

    def test_config_rails_with_values(self):
        """Test rails with actual values."""
        rails_config = {
            "input": {
                "flows": ["self check input"]
            },
            "output": {
                "flows": ["self check output"]
            },
            "models": [{
                "type": "main", "engine": "openai"
            }]
        }

        config = NemoGuardrailsConfig(rails=rails_config)

        assert config.rails == rails_config
        assert config.rails["input"]["flows"] == ["self check input"]
        assert config.rails["models"][0]["type"] == "main"

    def test_config_numeric_ranges(self):
        """Test that numeric fields accept reasonable ranges."""
        # Test max_retries
        config = NemoGuardrailsConfig(max_retries=0)
        assert config.max_retries == 0

        config = NemoGuardrailsConfig(max_retries=10)
        assert config.max_retries == 10

        # Test timeout_seconds
        config = NemoGuardrailsConfig(timeout_seconds=0.5)
        assert config.timeout_seconds == 0.5

        config = NemoGuardrailsConfig(timeout_seconds=300.0)
        assert config.timeout_seconds == 300.0
