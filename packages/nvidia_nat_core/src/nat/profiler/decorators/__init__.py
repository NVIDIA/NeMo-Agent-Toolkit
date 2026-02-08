# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nat.profiler.decorators.latency import LatencySensitivity
from nat.profiler.decorators.latency import latency_sensitive

__all__ = [
    "LatencySensitivity",
    "latency_sensitive",
]
