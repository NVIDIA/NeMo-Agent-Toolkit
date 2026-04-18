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

import importlib
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from nat.builder.llm import LLMProviderInfo
from nat.llm.oci_llm import OCIModelConfig
from nat.llm.oci_llm import oci_llm


@pytest.fixture(name="mock_builder")
def fixture_mock_builder():
    """Create a mock builder."""
    return MagicMock()


def test_oci_model_config_defaults():
    config = OCIModelConfig(model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")

    assert config.auth_type == "API_KEY"
    assert config.auth_profile == "DEFAULT"
    assert config.auth_file_location == "~/.oci/config"
    assert config.region == "us-chicago-1"
    assert config.endpoint == "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    assert config.context_size == 1024


def test_oci_model_config_derives_endpoint_from_region():
    config = OCIModelConfig(
        model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        region="eu-frankfurt-1",
    )

    assert config.endpoint == "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"


def test_oci_model_config_explicit_endpoint_overrides_region():
    config = OCIModelConfig(
        model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        region="eu-frankfurt-1",
        endpoint="https://custom.endpoint.example.com",
    )

    assert config.endpoint == "https://custom.endpoint.example.com"


def test_oci_model_config_accepts_endpoint_aliases():
    config = OCIModelConfig(
        model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        service_endpoint="https://custom.endpoint.example.com",
    )

    assert config.endpoint == "https://custom.endpoint.example.com"


@pytest.mark.asyncio
async def test_oci_llm_provider_yields_provider_info(mock_builder):
    config = OCIModelConfig(
        model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        region="us-chicago-1",
        compartment_id="ocid1.compartment.oc1..example",
    )

    async with oci_llm(config, mock_builder) as provider:
        assert isinstance(provider, LLMProviderInfo)
        assert provider.config is config
        assert "OCI" in provider.description


@patch.dict("os.environ", {}, clear=True)
def test_oci_model_config_does_not_depend_on_env():
    config = OCIModelConfig(model_name="nvidia/Llama-3.1-Nemotron-Nano-8B-v1")

    assert config.model_name == "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"


@patch("nat.cli.type_registry.GlobalTypeRegistry")
def test_oci_provider_registration(mock_global_registry):
    registry = MagicMock()
    mock_global_registry.get.return_value = registry

    sys.modules.pop("nat.llm.oci_llm", None)
    module = importlib.import_module("nat.llm.oci_llm")

    registry.register_llm_provider.assert_called_once()
    info = registry.register_llm_provider.call_args.args[0]
    assert info.config_type is module.OCIModelConfig
    assert info.build_fn is module.oci_llm
