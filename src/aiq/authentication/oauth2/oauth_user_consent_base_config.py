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

import logging
import secrets

from pydantic import Field
from pydantic import field_validator

from aiq.authentication.exceptions.auth_code_grant_exceptions import AuthCodeGrantConfigConsentPromptKeyFieldError
from aiq.authentication.oauth2.oauth_base_config import OAuthBaseConfig

logger = logging.getLogger(__name__)


class OAuthUserConsentConfigBase(OAuthBaseConfig):
    """
    Base OAuth 2.0 user consent authentication configuration model.
    Contains consent prompt-related fields for OAuth flows that require user interaction.
    Used for flows like Authorization Code Grant that need user consent.
    """
    consent_prompt_key: str = Field(description="The key used to retrieve the consent prompt location header, "
                                    " triggering the browser to complete the OAuth process from the front end.")
    consent_prompt_location_url: str | None = Field(
        default=None,
        description="302 redirect Location header to which the client will be redirected to the consent prompt.")

    state: str = Field(default=secrets.token_urlsafe(nbytes=16),
                       description="A URL-safe base64 format 16 byte random string",
                       frozen=True)

    @field_validator('consent_prompt_key')
    @classmethod
    def validate_consent_prompt_key(cls, value: str) -> str:
        """
        Validate consent prompt key for security.
        """
        if not value:
            raise AuthCodeGrantConfigConsentPromptKeyFieldError('value_missing',
                                                                'consent_prompt_key field value is required.')

        # Check for whitespace
        if len(value.strip()) != len(value):
            raise AuthCodeGrantConfigConsentPromptKeyFieldError(
                'whitespace_found', 'consent_prompt_key field value cannot have leading or trailing whitespace.')

        # Check for minimum length
        if len(value) < 8:
            raise AuthCodeGrantConfigConsentPromptKeyFieldError(
                'value_too_short',
                'consent_prompt_key field value must be at least 8 characters long for security. '
                'Got: {length} characters', {'length': len(value)})

        return value
