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
"""Pytest configuration and shared fixtures for OpenPipe ART tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def mock_art_library():
    """Automatically mock the ART library for all tests."""
    # Create mock ART module
    mock_art = MagicMock()

    # Mock ART classes
    mock_art.Backend = MagicMock()
    mock_art.TrainableModel = MagicMock()
    mock_art.Trajectory = MagicMock()
    mock_art.TrajectoryGroup = MagicMock()

    # Mock ART dev module
    mock_art.dev = MagicMock()
    mock_art.dev.InternalModelConfig = MagicMock()
    mock_art.dev.InitArgs = MagicMock()
    mock_art.dev.EngineArgs = MagicMock()
    mock_art.dev.TorchtuneArgs = MagicMock()
    mock_art.dev.TrainerArgs = MagicMock()
    mock_art.dev.OpenAIServerConfig = MagicMock()

    # Mock ART types
    mock_art.types = MagicMock()
    mock_art.types.TrainConfig = MagicMock()

    # Install the mock
    sys.modules['art'] = mock_art
    sys.modules['art.dev'] = mock_art.dev
    sys.modules['art.types'] = mock_art.types

    yield mock_art

    # Cleanup
    del sys.modules['art']
    del sys.modules['art.dev']
    del sys.modules['art.types']


@pytest.fixture(autouse=True)
def mock_openai_types():
    """Mock OpenAI types used in the code."""
    mock_openai = MagicMock()
    mock_openai.types = MagicMock()
    mock_openai.types.chat = MagicMock()
    mock_openai.types.chat.chat_completion = MagicMock()

    # Mock the Choice class
    mock_choice = MagicMock()
    mock_choice.return_value = MagicMock(index=0,
                                         logprobs=None,
                                         message={
                                             "role": "assistant", "content": "Test"
                                         },
                                         finish_reason="stop")
    mock_openai.types.chat.chat_completion.Choice = mock_choice

    sys.modules['openai'] = mock_openai
    sys.modules['openai.types'] = mock_openai.types
    sys.modules['openai.types.chat'] = mock_openai.types.chat
    sys.modules['openai.types.chat.chat_completion'] = mock_openai.types.chat.chat_completion

    yield mock_openai

    # Cleanup
    del sys.modules['openai']
    del sys.modules['openai.types']
    del sys.modules['openai.types.chat']
    del sys.modules['openai.types.chat.chat_completion']


@pytest.fixture
def disable_matplotlib():
    """Disable matplotlib for tests that don't need plotting."""
    import nat.plugins.openpipe.trainer as trainer_module
    original_value = trainer_module.MATPLOTLIB_AVAILABLE
    trainer_module.MATPLOTLIB_AVAILABLE = False
    yield
    trainer_module.MATPLOTLIB_AVAILABLE = original_value
