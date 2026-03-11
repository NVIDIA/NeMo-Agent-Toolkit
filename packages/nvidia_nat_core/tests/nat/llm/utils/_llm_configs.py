# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""LLM configuration models for testing."""

from pydantic import Field

from nat.data_models.llm import LLMBaseConfig
from nat.data_models.ssl_verification_mixin import SSLVerificationMixin


class LLMConfig(LLMBaseConfig):
    pass


class LLMConfigWithTimeout(LLMBaseConfig):
    request_timeout: float | None = Field(default=None, gt=0.0, description="HTTP request timeout in seconds.")


class LLMConfigWithTimeoutAndSSL(LLMConfigWithTimeout, SSLVerificationMixin):
    pass
