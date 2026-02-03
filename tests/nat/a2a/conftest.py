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
"""Shared fixtures for A2A tests."""

import pytest


class MockUserContext:
    """Mock user context for testing.

    This simple mock provides a user_id attribute for testing
    Context-dependent functionality without requiring full Context setup.
    """

    user_id: str = "test-user"


@pytest.fixture(name="mock_user_context")
def fixture_mock_user_context() -> MockUserContext:
    """Fixture providing a mock user context.

    Returns:
        MockUserContext with default test user ID
    """
    return MockUserContext()
