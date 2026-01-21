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

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import Request
from fastapi import Response

from nat.front_ends.fastapi.fastapi_front_end_config import FastApiFrontEndConfig

logger = logging.getLogger(__name__)


class VersioningOptions:
    """Options for version headers."""

    def __init__(self, *, api_version_header: bool = True, version: str = "1"):
        self.api_version_header = api_version_header
        self.version = version

    @classmethod
    def from_config(cls, cfg: FastApiFrontEndConfig):  # pragma: no cover - simple wiring helper
        version_cfg = getattr(cfg, "versioning", None)
        return cls(api_version_header=bool(getattr(version_cfg, "api_version_header", True)),
                   version=str(getattr(version_cfg, "version", 1)))


def apply_version_headers(response: Response, opts: VersioningOptions, *, is_legacy: bool = False) -> None:
    """Inject version header if enabled."""
    if opts.api_version_header:
        response.headers["X-API-Version"] = opts.version


def wrap_with_headers(handler: Callable, opts: VersioningOptions, *, is_legacy: bool) -> Callable:
    """Wrap a route handler to add version headers."""

    async def _wrapped(*args, **kwargs):
        response: Response | None = kwargs.get("response")
        # Some handlers accept Response as first positional arg; detect by type
        if response is None:
            for arg in args:
                if isinstance(arg, Response):
                    response = arg
                    break

        result = await handler(*args, **kwargs)
        if response is not None:
            apply_version_headers(response, opts, is_legacy=is_legacy)
        return result

    return _wrapped


def is_versioned_request(request: Request) -> bool:
    """Quick check to detect /v{n}/ prefix on the path."""
    return request.url.path.startswith("/v")
