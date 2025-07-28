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


class AuthCodeGrantFlowError(Exception):
    """Raised when Auth Code Grant Flow fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantFlowRefreshTokenError(Exception):
    """Raised when Auth Code Grant Flow requesting access token using refresh flow fails unexpectedly. """

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigClientServerUrlFieldError(Exception):
    """Raised when Auth Code Grant Config client server URL field validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigAuthorizationUrlFieldError(Exception):
    """Raised when Auth Code Grant Config authorization_url and authorization_token_url field validation fails."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigConsentPromptKeyFieldError(Exception):
    """Raised when Auth Code Grant Config consent_prompt_key field validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigClientSecretFieldError(Exception):
    """Raised when Auth Code Grant Config client_secret field validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigClientIDFieldError(Exception):
    """Raised when Auth Code Grant Config client_id field validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigScopeFieldError(Exception):
    """Raised when Auth Code Grant Config scope field validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class AuthCodeGrantConfigAudienceFieldError(Exception):
    """Raised when Auth Code Grant Config audience field validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)
