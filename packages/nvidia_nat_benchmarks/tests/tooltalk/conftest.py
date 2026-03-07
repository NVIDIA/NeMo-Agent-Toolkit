# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conftest for ToolTalk tests.

ToolTalk's utils.py unconditionally imports torch and transformers for
semantic string comparison (HFVectorizer), but these are never used during
evaluation. We mock them out so tests can run without installing PyTorch.
"""

import sys
import types
from unittest.mock import MagicMock


def _ensure_mock_module(name: str):
    """Add a mock module to sys.modules if not already importable."""
    if name not in sys.modules:
        try:
            __import__(name)
        except (ImportError, ModuleNotFoundError):
            sys.modules[name] = MagicMock()


# Mock torch and transformers before any tooltalk imports trigger them
_ensure_mock_module("torch")
_ensure_mock_module("transformers")
