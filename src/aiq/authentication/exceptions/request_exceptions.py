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


class BaseUrlValidationError(Exception):
    """Raised when HTTP Base URL validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class HTTPMethodValidationError(Exception):
    """Raised when HTTP Method validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class QueryParameterValidationError(Exception):
    """Raised when HTTP Query Parameter validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class HTTPHeaderValidationError(Exception):
    """Raised when HTTP Header validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)


class BodyValidationError(Exception):
    """Raised when HTTP Body validation fails unexpectedly."""

    def __init__(self, error_code: str, message: str, *args):
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}", *args)
