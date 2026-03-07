# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conftest for BFCL tests — mocks optional dependencies if missing."""

import sys
from unittest.mock import MagicMock


def _ensure_mock_module(name: str):
    if name not in sys.modules:
        try:
            __import__(name)
        except (ImportError, ModuleNotFoundError, TypeError):
            sys.modules[name] = MagicMock()


# bfcl may try to import optional deps
_ensure_mock_module("torch")
_ensure_mock_module("transformers")
_ensure_mock_module("tree_sitter")
_ensure_mock_module("tree_sitter_java")
_ensure_mock_module("tree_sitter_javascript")
