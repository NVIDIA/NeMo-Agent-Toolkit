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

import re

from pydantic import BaseModel
from pydantic import Field

from nat.data_models.gated_field_mixin import GatedFieldMixin

# The system prompt format for thinking is different for these, so we need to distinguish them here with two separate
# regex patterns
_NVIDIA_NEMOTRON_REGEX = re.compile(r"^nvidia/nvidia.*nemotron", re.IGNORECASE)
_LLAMA_NEMOTRON_REGEX = re.compile(r"^nvidia/llama.*nemotron", re.IGNORECASE)
_MODEL_KEYS = ("model_name", "model", "azure_deployment")


class ThinkingMixin(
        BaseModel,
        GatedFieldMixin,
        field_name="thinking",
        default_if_supported=None,
        keys=_MODEL_KEYS,
        supported=(_NVIDIA_NEMOTRON_REGEX, _LLAMA_NEMOTRON_REGEX),
):
    """
    Mixin class for thinking configuration. Only supported on Nemotron models.

    Attributes:
        thinking: Whether to enable thinking. Defaults to None when supported on the model.
    """
    thinking: bool | None = Field(
        default=None,
        description="Whether to enable thinking. Defaults to None when supported on the model.",
        exclude=True,
    )

    @property
    def thinking_system_prompt(self) -> str | None:
        """
        Returns the system prompt to use for thinking.
        For NVIDIA Nemotron, returns "/think" if enabled, else "/no_think".
        For Llama Nemotron, returns "detailed thinking on" if enabled, else "detailed thinking off".
        If thinking is not supported on the model, returns None.

        Returns:
            str | None: The system prompt to use for thinking.
        """
        if self.thinking is None:
            return None
        for key in _MODEL_KEYS:
            if hasattr(self, key):
                if _NVIDIA_NEMOTRON_REGEX.match(getattr(self, key)):
                    return "/think" if self.thinking else "/no_think"
                elif _LLAMA_NEMOTRON_REGEX.match(getattr(self, key)):
                    return f"detailed thinking {'on' if self.thinking else 'off'}"
