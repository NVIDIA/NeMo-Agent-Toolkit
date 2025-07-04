# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.credentials_manager import _CredentialsManager
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.authentication.oauth2.oauth_user_consent_base_config import OAuthUserConsentConfigBase
from aiq.data_models.authentication import AuthenticationBaseConfig
from aiq.data_models.config import AIQConfig
from aiq.data_models.config import GeneralConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from aiq.test.functions import EchoFunctionConfig


@pytest.fixture
async def auth_configs() -> dict[str, AuthenticationBaseConfig]:
    """Load authentication configs from config.yml file."""
    config_path = Path(__file__).parent / "config.yml"

    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    configs: dict[str, AuthenticationBaseConfig] = {}

    for config_name, config_data in yaml_data.get("authentication", {}).items():
        config_type = config_data.pop("_type")

        if config_type == "oauth2_authorization_code_grant":
            configs[config_name] = AuthCodeGrantConfig(**config_data)
        elif config_type == "api_key":
            configs[config_name] = APIKeyConfig(**config_data)
        else:
            raise ValueError(f"Unknown config type: {config_type}")

    return configs


@pytest.fixture
async def minimum_configured_cors_app() -> FastAPI:
    """Create a FastAPI app with minimal CORS configuration for testing."""
    # Define minimal CORS config
    minimal_cors_config = FastApiFrontEndConfig.CrossOriginResourceSharing(
        allow_origins=["http://localhost:3000"],
        allow_headers=["Content-Type", "Authorization"],
        allow_methods=["GET", "POST", "OPTIONS"],
    )
    # Create AIQConfig to create fastapi app with configured CORS
    aiq_config = AIQConfig(general=GeneralConfig(front_end=FastApiFrontEndConfig(cors=minimal_cors_config)),
                           workflow=EchoFunctionConfig())

    # Create FastAPI
    worker = FastApiFrontEndPluginWorker(aiq_config)
    app = FastAPI()
    worker.set_cors_config(app)

    # Add test routes for all methods
    @app.get("/test")
    @app.post("/test")
    @app.options("/test")
    async def test_endpoint():
        return {"message": "success"}

    return app


async def test_credential_manager_singleton():
    """Test that the credential manager is a singleton class."""

    credentials1 = _CredentialsManager()
    credentials2 = _CredentialsManager()

    assert credentials1 is credentials2


async def test_validate_unique_consent_prompt_keys_valid(auth_configs: dict[str, AuthenticationBaseConfig]):
    """Test that a RuntimeError is NOT raised for unique consent prompt keys."""

    configs = auth_configs.copy()

    # Filter to only include OAuthUserConsentConfigBase configs (configs with consent_prompt_key)
    oauth_consent_configs: dict[str, AuthenticationBaseConfig] = {
        name: config
        for name, config in configs.items() if isinstance(config, OAuthUserConsentConfigBase)
    }

    # Add an additional OAuth config with unique consent prompt key to test uniqueness validation
    oauth_config2 = AuthCodeGrantConfig(client_server_url="http://localhost:8001",
                                        authorization_url="https://auth2.example.com/oauth/authorize",
                                        authorization_token_url="https://auth2.example.com/oauth/token",
                                        consent_prompt_key="unique_key_2_different_from_yaml",
                                        client_secret="another_secure_client_secret_123456",
                                        client_id="my_other_client_app_123",
                                        audience="api2.example.com",
                                        scope=["read"])

    oauth2_key = "oauth2"
    oauth_consent_configs[oauth2_key] = oauth_config2

    # This should not raise any exception
    _CredentialsManager().validate_unique_consent_prompt_keys(oauth_consent_configs)


async def test_validate_unique_consent_prompt_keys_invalid(auth_configs: dict[str, AuthenticationBaseConfig]) -> None:
    """Test that a RuntimeError IS raised for duplicate consent prompt keys."""

    configs = auth_configs.copy()

    # Filter to only include OAuthUserConsentConfigBase configs (configs with consent_prompt_key)
    oauth_consent_configs: dict[str, AuthenticationBaseConfig] = {
        name: config
        for name, config in configs.items() if isinstance(config, OAuthUserConsentConfigBase)
    }

    # Get the consent prompt key from an existing OAuth config
    if oauth_consent_configs:
        first_oauth_config = next(iter(oauth_consent_configs.values()))

        if isinstance(first_oauth_config, OAuthUserConsentConfigBase):
            duplicate_key = first_oauth_config.consent_prompt_key
        else:
            duplicate_key = "mock_consent_prompt_key_secure"
    else:
        duplicate_key = "mock_consent_prompt_key_secure"

    # Add another OAuth config with the same consent prompt key to trigger the error
    oauth_config_duplicate = AuthCodeGrantConfig(
        client_server_url="http://localhost:8001",
        authorization_url="https://auth2.example.com/oauth/authorize",
        authorization_token_url="https://auth2.example.com/oauth/token",
        consent_prompt_key=duplicate_key,  # Using same key as existing config
        client_secret="another_secure_client_secret_123456",
        client_id="my_other_client_app_123",
        audience="api2.example.com",
        scope=["read"])

    duplicate_oauth_key = "duplicate_oauth"
    oauth_consent_configs[duplicate_oauth_key] = oauth_config_duplicate

    # This should raise a RuntimeError
    with pytest.raises(RuntimeError):
        _CredentialsManager().validate_unique_consent_prompt_keys(oauth_consent_configs)


async def test_cors_config_valid_requests(minimum_configured_cors_app: FastAPI):
    """Test valid CORS configurations and requests that should be accepted."""

    # Define valid test data (allowed by our minimal CORS config)
    valid_origins = ["http://localhost:3000"]
    valid_headers = ["Content-Type", "Authorization"]
    valid_methods = ["GET", "POST", "OPTIONS"]

    client = TestClient(minimum_configured_cors_app)

    # Test invalid origins (should be accepted)
    for origin in valid_origins:
        for method in valid_methods:
            for header in valid_headers:
                response = client.request(method, "/test", headers={"Origin": origin, header: "test"})

                # CORS should accept requests from valid origins
                assert response.status_code == 200, \
                    f"Expected 200 but got {response.status_code} for {origin} {method} {header}"


async def test_cors_config_invalid_requests(minimum_configured_cors_app: FastAPI):
    """Test invalid CORS configurations and requests that should be rejected."""

    # Define invalid test data (not allowed by our minimal CORS config)
    invalid_origins = ["http://localhost:8080", "https://evil.com"]
    invalid_headers = ["X-Custom-Header", "X-Admin-Token"]
    invalid_methods = ["DELETE", "PUT", "PATCH"]

    client = TestClient(minimum_configured_cors_app)

    # Test invalid origins (should be rejected)
    for invalid_origin in invalid_origins:
        for invalid_method in invalid_methods:
            for invalid_header in invalid_headers:
                response = client.request(invalid_method,
                                          "/test",
                                          headers={
                                              "Origin": invalid_origin, invalid_header: "test"
                                          })

                # CORS should reject requests from invalid origins
                assert response.status_code != 200, \
                    f"Expected rejection but got {response.status_code} for invalid origin {invalid_origin}"
