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

from nat.data_models.model_gated_field_mixin import ModelGatedFieldMixin

_UNSUPPORTED_TOP_P_MODELS = (re.compile(r"gpt-?5", re.IGNORECASE), )


class TopPMixin(
        BaseModel,
        ModelGatedFieldMixin[float],
        field_name="top_p",
        default_if_supported=1.0,
        unsupported_models=_UNSUPPORTED_TOP_P_MODELS,
):
    """
    Mixin class for top-p configuration.
    """
    top_p: float | None = Field(default=None, ge=0.0, le=1.0, description="Top-p for distribution sampling.")
