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
"""Function intercept implementations for NeMo Agent Toolkit."""

from nat.intercepts.cache_intercept import CacheIntercept
from nat.intercepts.function_intercept import CallNext
from nat.intercepts.function_intercept import CallNextStream
from nat.intercepts.function_intercept import FunctionIntercept
from nat.intercepts.function_intercept import FunctionInterceptChain
from nat.intercepts.function_intercept import FunctionInterceptContext
from nat.intercepts.function_intercept import validate_intercepts

__all__ = [
    "CacheIntercept",
    "CallNext",
    "CallNextStream",
    "FunctionIntercept",
    "FunctionInterceptChain",
    "FunctionInterceptContext",
    "validate_intercepts",
]
