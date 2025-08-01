# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import typing

from .common import BaseModelRegistryTag
from .common import TypedBaseModel


class GuardrailsBaseConfig(TypedBaseModel, BaseModelRegistryTag):
    """Base configuration for guardrails implementations."""
    pass


GuardrailsBaseConfigT = typing.TypeVar("GuardrailsBaseConfigT", bound=GuardrailsBaseConfig)
