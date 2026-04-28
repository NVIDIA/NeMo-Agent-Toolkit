# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os

from nat.data_models.common import SerializableSecretStr
from nat.data_models.common import get_secret_value
from tavily import AsyncTavilyClient


def build_async_client(api_key: SerializableSecretStr | None) -> AsyncTavilyClient:
    """Construct an AsyncTavilyClient, resolving the API key from config or TAVILY_API_KEY env."""
    resolved = None
    if api_key:
        api_key_value = get_secret_value(api_key)
        if api_key_value:
            api_key_value = api_key_value.strip()
            if api_key_value:
                resolved = api_key_value

    if not resolved:
        env_api_key = os.environ.get("TAVILY_API_KEY")
        if env_api_key:
            env_api_key = env_api_key.strip()
            if env_api_key:
                resolved = env_api_key

    if not resolved:
        raise ValueError(
            "Tavily API key not provided. Set the `api_key` config field or the TAVILY_API_KEY env var.")
    return AsyncTavilyClient(api_key=resolved, client_name="nvidia-nat-tavily")
