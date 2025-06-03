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


class OAuthCodeFlowError(Exception):
    """Raised when OAuth2.0 Code flow fails unexpectedly."""
    pass


class OAuthRefreshTokenError(Exception):
    """Raised when OAuth2.0 requesting access token using refresh flow fails unexpectedly. """
    pass


class BaseUrlValidationError(Exception):
    """Raised when HTTP Base URL validation fails unexpectedly."""
    pass


class HTTPMethodValidationError(Exception):
    """Raised when HTTP Method validation fails unexpectedly."""
    pass


class QueryParameterValidationError(Exception):
    """Raised when HTTP Query Parameter validation fails unexpectedly."""
    pass


class HTTPHeaderValidationError(Exception):
    """Raised when HTTP Header validation fails unexpectedly."""
    pass


class BodyValidationError(Exception):
    """Raised when HTTP Body validation fails unexpectedly."""
    pass


class APIRequestError(Exception):
    """Raised when making an API request fails unexpectedly."""
    pass
