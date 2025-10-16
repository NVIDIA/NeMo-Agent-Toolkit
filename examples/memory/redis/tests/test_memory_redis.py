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

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow(redis_server: dict[str, str | int], phoenix_trace_url: str, examples_dir: Path):
    from nat.plugins.redis.memory import RedisMemoryClientConfig
    from nat.runtime.loader import load_config
    from nat.test.utils import run_workflow

    config_file = (examples_dir / "memory/redis/configs/config.yml")

    config = load_config(config_file)
    config.general.telemetry.tracing["phoenix"].endpoint = phoenix_trace_url

    existing_redis_config = config.memory['redis_memory']
    redis_config = RedisMemoryClientConfig(host=redis_server["host"],
                                           port=redis_server["port"],
                                           db=redis_server["db"],
                                           key_prefix=existing_redis_config.key_prefix,
                                           embedder=existing_redis_config.embedder)

    config.memory['redis_memory'] = redis_config
    await run_workflow(config=config, question="my favorite flavor is strawberry", expected_answer="strawberry")
    await run_workflow(config=config, question="what flavor of ice-cream should I get?", expected_answer="strawberry")
