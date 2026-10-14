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
"""Shared fixtures for the security middleware tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any

import pytest

from nat.data_models.api_server import ChatResponseChunk


@pytest.fixture(name="structured_chunk_stream")
def fixture_structured_chunk_stream() -> Callable[[Any], AsyncIterator[ChatResponseChunk]]:
    """A ``call_next`` stream that yields ``ChatResponseChunk`` objects whose text joins to "Hello world"."""

    async def _stream(_value: Any) -> AsyncIterator[ChatResponseChunk]:
        yield ChatResponseChunk.create_streaming_chunk("Hello ", id_="echo")
        yield ChatResponseChunk.create_streaming_chunk("world", id_="echo")

    return _stream
