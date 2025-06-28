# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import base64
import enum
import logging
import os
import typing
from copy import deepcopy

from cryptography.fernet import Fernet

from aiq.authentication.oauth2.auth_code_grant_config import AuthCodeGrantConfig
from aiq.builder.context import Singleton
from aiq.data_models.authentication import AuthenticationBaseConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig
from aiq.front_ends.fastapi.fastapi_front_end_config import FrontEndBaseConfig

if (typing.TYPE_CHECKING):
    from aiq.data_models.config import AIQConfig

logger = logging.getLogger(__name__)


class _CredentialsManager(metaclass=Singleton):

    def __init__(self):
        """
        Credentials Manager to store AIQ Authorization configurations.
        """
        super().__init__()
        self._authentication_configs: dict[str, AuthenticationBaseConfig] = {}
        self._swap_flag: bool = True
        self._encrypt_flag: bool = True
        self._get_encrypt_key_flag = True
        self._full_config: "AIQConfig" = None
        self._oauth_credentials_flag: asyncio.Event = asyncio.Event()
        self._consent_prompt_flag: asyncio.Event = asyncio.Event()
        self._credentials_encryption_key: Fernet | None = None

    def swap_authentication_configs(self, authentication_configs: dict[str, AuthenticationBaseConfig]) -> None:
        """
        Transfer ownership of the sensitive AIQ Authorization configuration attributes to the
        CredentialsManager.

        Args:
            authentication_configs (dict[str, AuthenticationBaseConfig]): Dictionary of registered authentication
            configs.
        """
        if self._swap_flag:
            self._authentication_configs = deepcopy(authentication_configs)
            authentication_configs.clear()
            self._swap_flag = False

    def validate_unique_consent_prompt_keys(self) -> None:
        """
        Validate that all AuthCodeGrantConfig instances have unique consent_prompt_key values.

        Raises:
            RuntimeError: If duplicate consent prompt keys are found.
        """
        consent_prompt_keys: list[str] = []

        # Collect all consent prompt keys and their associated config names
        for _, auth_config in self._authentication_configs.items():
            if isinstance(auth_config, AuthCodeGrantConfig):

                if auth_config.consent_prompt_key in consent_prompt_keys:
                    error_message = (f"Duplicate consent_prompt_key found: {auth_config.consent_prompt_key}. "
                                     "Please ensure consent_prompt_key is unique across Authentication configs.")
                    logger.critical(error_message)
                    raise RuntimeError('duplicate_consent_prompt_key', error_message)
                else:
                    consent_prompt_keys.append(auth_config.consent_prompt_key)

    def encrypt_authentication_configs(self) -> None:
        """
        Encrypts Authentication Configurations.
        """
        if self._encrypt_flag:
            for model in self._authentication_configs.values():
                for field in model.model_fields:
                    field_value = getattr(model, field)

                    # Only encrypt string values, because they are the only ones that are sensitive currently.
                    if isinstance(field_value, str) and field != "type" and not isinstance(field_value, enum.Enum):
                        encrypted_value = self.encrypt_value(field_value)
                        setattr(model, field, encrypted_value)

            self._encrypt_flag = False

    def generate_credentials_encryption_key(self) -> None:
        """
        Generate an encryption key for the Authentication Credentials.
        """
        if self._get_encrypt_key_flag:
            os.environ["CREDENTIALS_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
            self._credentials_encryption_key = Fernet(os.environ["CREDENTIALS_ENCRYPTION_KEY"].encode())
            self._get_encrypt_key_flag = False

    def encrypt_value(self, un_encrypted_value: str) -> str:
        """
        Encrypts a sensitive string value and returns a Base64-encoded string.

        Args:
            un_encrypted_value: Sensitive string value to be encrypted.

        Returns:
            str: Base64-encoded encrypted string.
        """

        if self._credentials_encryption_key is None:
            raise RuntimeError("Encryption key not set. Initialize with a valid Fernet key.")

        # Encrypt the value.
        encrypted_bytes: bytes = self._credentials_encryption_key.encrypt(un_encrypted_value.encode())

        # Return the encrypted value as a Base64-encoded string.
        return base64.urlsafe_b64encode(encrypted_bytes).decode()

    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypts a Base64-encoded encrypted string.

        Args:
            encrypted_value: Base64-encoded string to decrypt.

        Returns:
            str: Decrypted original string.
        """

        if self._credentials_encryption_key is None:
            raise RuntimeError("Encryption key not set. Initialize with a valid Fernet key.")

        # Decrypt the value.
        encrypted_bytes: bytes = base64.urlsafe_b64decode(encrypted_value.encode())

        # Return the decrypted value as a string.
        return self._credentials_encryption_key.decrypt(encrypted_bytes).decode()

    def get_authentication_config(self, authentication_config_name: str | None) -> AuthenticationBaseConfig | None:
        """Retrieve the stored authentication config by registered name."""

        if authentication_config_name not in self._authentication_configs:
            logger.error("Authentication config not found: %s", authentication_config_name)
            return None

        return self._authentication_configs.get(authentication_config_name)

    def get_authentication_config_by_state(self, state: str) -> AuthCodeGrantConfig | None:
        """Retrieve the stored authentication config by state."""

        for _, authentication_config in self._authentication_configs.items():
            if isinstance(authentication_config, AuthCodeGrantConfig):
                if self.decrypt_value(authentication_config.state) == state:
                    return authentication_config

        logger.error("Authentication config not found by the provided state.")
        return None

    def get_authentication_config_name(self, authentication_config: AuthenticationBaseConfig) -> str | None:
        """Retrieve the stored authentication config name."""

        for registered_config_name, registered_config in self._authentication_configs.items():
            if (authentication_config == registered_config):
                return registered_config_name

        logger.error("Authentication config name not found by the provided authentication config model.")
        return None

    def get_authentication_config_by_consent_prompt_key(self, consent_prompt_key: str) -> AuthCodeGrantConfig | None:
        """Retrieve the stored authentication config by consent prompt key."""
        for _, authentication_config in self._authentication_configs.items():
            if isinstance(authentication_config, AuthCodeGrantConfig):
                if self.decrypt_value(authentication_config.consent_prompt_key) == consent_prompt_key:
                    return authentication_config
        return None

    def validate_and_set_cors_config(self, front_end_config: FrontEndBaseConfig) -> None:
        """
        Validate and set the CORS authentication configuration for the frontend.
        """

        default_allow_origins: list[str] = ["http://localhost:3000"]
        default_allow_headers: list[str] = ["Content-Type", "Authorization"]
        default_allow_methods: list[str] = ["POST", "OPTIONS"]

        try:
            if not isinstance(front_end_config, FastApiFrontEndConfig):
                raise ValueError("Configuration is not of type FastApiFrontEndConfig.")

            # Allow the AIQ frontend browser to access the OAuth server in headless execution modes.
            if front_end_config.cors.allow_origins is None:
                front_end_config.cors.allow_origins = default_allow_origins
            else:
                for item in default_allow_origins:
                    if item not in front_end_config.cors.allow_origins:
                        front_end_config.cors.allow_origins.append(item)

            # Allow minimum headers to access the OAuth server in headless execution modes.
            if front_end_config.cors.allow_headers is None:
                front_end_config.cors.allow_headers = default_allow_headers
            else:
                for item in default_allow_headers:
                    if item not in front_end_config.cors.allow_headers:
                        front_end_config.cors.allow_headers.append(item)

            # Allow minimum methods to access the OAuth server in headless execution modes.
            if front_end_config.cors.allow_methods is None:
                front_end_config.cors.allow_methods = default_allow_methods
            else:
                for item in default_allow_methods:
                    if item not in front_end_config.cors.allow_methods:
                        front_end_config.cors.allow_methods.append(item)

            _CredentialsManager().full_config.general.front_end = front_end_config

        except ValueError:
            _CredentialsManager().full_config.general.front_end = FastApiFrontEndConfig(
                cors=FastApiFrontEndConfig.CrossOriginResourceSharing(
                    allow_origins=default_allow_origins,
                    allow_headers=default_allow_headers,
                    allow_methods=default_allow_methods,
                ))

    def get_registered_authentication_count(self) -> int:
        """
        Get the number of registered authentication configs.
        """
        return len(self._authentication_configs)

    async def wait_for_oauth_credentials(self) -> None:
        """
        Block until the oauth credentials are set in the redirect uri.
        """
        await self._oauth_credentials_flag.wait()

    async def set_oauth_credentials(self):
        """
        Unblock until the oauth credentials are set in the redirect uri.
        """
        self._oauth_credentials_flag.set()

    async def wait_for_consent_prompt_url(self):
        """
        Block until the consent prompt location header has been retrieved.
        """
        await self._consent_prompt_flag.wait()

    async def set_consent_prompt_url(self):
        """
        Unblock until the consent prompt location header has been retrieved.
        """
        self._consent_prompt_flag.set()

    @property
    def full_config(self) -> "AIQConfig":
        """Get the loaded AIQConfig."""
        return self._full_config

    @full_config.setter
    def full_config(self, full_config: "AIQConfig") -> None:
        """Set the loaded AIQConfig."""
        self._full_config = full_config
