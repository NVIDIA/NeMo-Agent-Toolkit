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
"""Exceptions raised by the NeMo Guardrails middleware."""


class PostInvokeBlockedError(RuntimeError):
    """Raised by ``on_post_invoke_blocked`` when a post_invoke rail blocks the function output.

    Carries the rail's message so callers can surface it or re-raise as appropriate.
    Override ``GuardrailsMiddleware.on_post_invoke_blocked`` to return a value instead.
    """

    def __init__(self, message: str) -> None:
        self.block_message: str = message
        super().__init__(f"Post-invoke rail blocked: {message!r}")


__all__ = ["PostInvokeBlockedError"]
