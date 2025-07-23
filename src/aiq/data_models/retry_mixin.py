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

from pydantic import BaseModel
from pydantic import Field


class RetryMixin(BaseModel):
    """Mixin class for retry configuration."""
    do_auto_retry: bool = Field(default=True,
                                description="Whether to automatically retry method calls"
                                " that fail with a retryable error.")
    num_retries: int = Field(default=5,
                             description="Number of times to retry a method call that fails"
                             " with a retryable error.")
    retry_on_status_codes: list[int | str] = Field(default_factory=lambda: [429, 500, 502, 503, 504],
                                                   description="List of HTTP status codes that should trigger a retry.")
    retry_on_errors: list[str] | None = Field(default_factory=lambda: ["Too Many Requests"],
                                              description="List of error substrings that should trigger a retry.")
