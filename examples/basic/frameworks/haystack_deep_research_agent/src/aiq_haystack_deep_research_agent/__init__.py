# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Re-export pipelines helpers for convenience
try:
	from .pipelines.search import create_search_tool  # noqa: F401
	from .pipelines.rag import create_rag_tool  # noqa: F401
	from .pipelines.indexing import run_startup_indexing  # noqa: F401
except Exception:  # pragma: no cover - optional during install time
	pass