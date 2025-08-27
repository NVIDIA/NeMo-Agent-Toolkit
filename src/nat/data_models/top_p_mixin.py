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


class TopPMixin(
        BaseModel,
        GatedFieldMixin,
        field_name="top_p",
        default_if_supported=1.0,
        keys=("model_name", "model", "azure_deployment"),
        unsupported=(re.compile(r"gpt-?5", re.IGNORECASE), ),
):
    """
    Mixin class for top-p configuration. Unsupported on models like gpt-5.

    Attributes:
        top_p: Top-p for distribution sampling. Defaults to 1.0 when supported on the model.
    """
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Top-p for distribution sampling. Defaults to 1.0 when supported on the model.",
    )
