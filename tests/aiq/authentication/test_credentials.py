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

import enum
import os
from pathlib import Path

import pytest

from aiq.authentication.api_key.api_key_config import APIKeyConfig
from aiq.authentication.credentials_manager import _CredentialsManager
from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.data_models.config import AIQConfig
from aiq.utils.data_models.schema_validator import validate_schema


async def test_credential_manager_singleton():
    """Test that the credential manager is a singleton class."""

    credentials1 = _CredentialsManager()
    credentials2 = _CredentialsManager()

    assert credentials1 is credentials2


async def test_credential_persistence():
    """Test that the credential manager can swap authorization configuration and persist credentials."""

    config_dict = load_and_override_config(Path("tests/aiq/authentication/config.yml"), overrides=())

    config = validate_schema(config_dict, AIQConfig)

    # Swap credentials and ensure they are not the same.
    assert _CredentialsManager().get_authentication_config("jira") != config.authentication.get("jira")
    assert not config.authentication

    # Ensure credentials can only be swapped once.
    assert _CredentialsManager().get_authentication_config("jira") != config.authentication.get("jira")

    # Ensure None is returned if the provider does not exist.
    test = _CredentialsManager().get_authentication_config("invalid_provider")
    assert test is None


def _verify_encryption_for_config(original_config, encrypted_config):
    """Helper function to verify that a config instance has been properly encrypted."""\

    # Loop through all model fields for this config
    for field_name in encrypted_config.model_fields:

        if field_name != "type":

            original_field_value = getattr(original_config, field_name)
            encrypted_field_value = getattr(encrypted_config, field_name)

            if isinstance(original_field_value, str) and not isinstance(original_field_value, enum.Enum):

                # Verify that the original value does not match the encrypted value, raise an error if they are alike.
                assert original_field_value != encrypted_field_value, (
                    f"Field {encrypted_field_value} was not encrypted!")

                # Verify that we can decrypt back to the original value, raise an error if the values differ.
                decrypted_field_value = _CredentialsManager().decrypt_value(encrypted_field_value)
                assert decrypted_field_value == original_field_value, (
                    f"Field {encrypted_field_value} did not decrypt to original value!")


async def test_encrypt_authentication_configs():
    """Test that the encrypt_authentication_configs method properly encrypts all authentication configuration fields."""

    # Create original authentication config instances
    original_auth_code_grant = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                                   authorization_url="https://auth.example.com/oauth/authorize",
                                                   authorization_token_url="https://auth.example.com/oauth/token",
                                                   consent_prompt_key="oauth_consent_key",
                                                   client_secret="super_secure_client_secret_123456",
                                                   client_id="my_client_app_123",
                                                   audience="api.example.com",
                                                   scope=["read", "write"])

    original_api_key = APIKeyConfig(api_key="test_api_key_12345", header_name="Authorization", header_prefix="Bearer")

    # Reset the singleton state to allow encryption operations to run
    credentials_manager = _CredentialsManager()
    credentials_manager._encrypt_flag = True
    credentials_manager._get_encrypt_key_flag = True
    credentials_manager._credentials_encryption_key = None

    # Set up credentials manager with copies of the configs to be encrypted.
    credentials_manager._authentication_configs = {
        "auth_code_grant_config": original_auth_code_grant.model_copy(),
        "api_key_config": original_api_key.model_copy()
    }

    # Generate encryption key and encrypt configs
    credentials_manager.generate_credentials_encryption_key()
    credentials_manager.encrypt_authentication_configs()

    # Verify encryption for each config type using helper function
    encrypted_auth_code_grant = credentials_manager._authentication_configs["auth_code_grant_config"]
    encrypted_api_key = credentials_manager._authentication_configs["api_key_config"]

    _verify_encryption_for_config(original_auth_code_grant, encrypted_auth_code_grant)
    _verify_encryption_for_config(original_api_key, encrypted_api_key)

    # Clean up environment variable to prevent interference with other tests
    if "CREDENTIALS_ENCRYPTION_KEY" in os.environ:
        del os.environ["CREDENTIALS_ENCRYPTION_KEY"]


def test_validate_unique_consent_prompt_keys_valid():
    """Test that a RuntimeError is NOT raised for duplicate consent prompt keys."""

    oauth_config1 = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                        authorization_url="https://auth.example.com/oauth/authorize",
                                        authorization_token_url="https://auth.example.com/oauth/token",
                                        consent_prompt_key="unique_key_1",
                                        client_secret="super_secure_client_secret_123456",
                                        client_id="my_client_app_123",
                                        audience="api.example.com",
                                        scope=["read", "write"])

    oauth_config2 = AuthCodeGrantConfig(client_server_url="http://localhost:8001",
                                        authorization_url="https://auth2.example.com/oauth/authorize",
                                        authorization_token_url="https://auth2.example.com/oauth/token",
                                        consent_prompt_key="unique_key_2",
                                        client_secret="another_secure_client_secret_123456",
                                        client_id="my_other_client_app_123",
                                        audience="api2.example.com",
                                        scope=["read"])

    api_config = APIKeyConfig(api_key="test_api_key_12345", header_name="Authorization", header_prefix="Bearer")

    # Set up configs with unique consent prompt keys
    _CredentialsManager()._authentication_configs = {
        "oauth1": oauth_config1, "oauth2": oauth_config2, "api": api_config
    }

    # This should not raise any exception
    _CredentialsManager().validate_unique_consent_prompt_keys()


def test_validate_unique_consent_prompt_keys_invalid():
    """Test that a RuntimeError is NOT raised for duplicate consent prompt keys."""

    oauth_config1 = AuthCodeGrantConfig(client_server_url="http://localhost:8000",
                                        authorization_url="https://auth.example.com/oauth/authorize",
                                        authorization_token_url="https://auth.example.com/oauth/token",
                                        consent_prompt_key="unique_key_1",
                                        client_secret="super_secure_client_secret_123456",
                                        client_id="my_client_app_123",
                                        audience="api.example.com",
                                        scope=["read", "write"])

    oauth_config2 = AuthCodeGrantConfig(client_server_url="http://localhost:8001",
                                        authorization_url="https://auth2.example.com/oauth/authorize",
                                        authorization_token_url="https://auth2.example.com/oauth/token",
                                        consent_prompt_key="unique_key_1",
                                        client_secret="another_secure_client_secret_123456",
                                        client_id="my_other_client_app_123",
                                        audience="api2.example.com",
                                        scope=["read"])

    api_config = APIKeyConfig(api_key="test_api_key_12345", header_name="Authorization", header_prefix="Bearer")

    # Set up configs with unique consent prompt keys
    _CredentialsManager()._authentication_configs = {
        "oauth1": oauth_config1, "oauth2": oauth_config2, "api": api_config
    }

    # This should rasie a RuntimeError
    with pytest.raises(RuntimeError):
        _CredentialsManager().validate_unique_consent_prompt_keys()
