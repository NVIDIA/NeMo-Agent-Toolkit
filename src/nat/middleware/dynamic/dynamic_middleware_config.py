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
"""Configuration for dynamic middleware."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

from nat.data_models.component import ComponentGroup
from nat.data_models.component_ref import AuthenticationRef
from nat.data_models.component_ref import EmbedderRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.component_ref import MemoryRef
from nat.data_models.component_ref import ObjectStoreRef
from nat.data_models.component_ref import RetrieverRef
from nat.data_models.middleware import FunctionMiddlewareBaseConfig


class AllowedComponentFunctions(BaseModel):
    """Component functions allowed for auto-registration.

    Default allowlists are provided for each component type. User-provided
    values are automatically merged with defaults.
    Set to None or omit to use only defaults.
    """

    llms: set[str] | None = Field(
        default=None, description="Additional LLM functions that should be allowed to register with middleware.")
    embedders: set[str] | None = Field(
        default=None, description="Additional Embedder functions that should be allowed to register with middleware.")
    retrievers: set[str] | None = Field(
        default=None, description="Additional Retriever functions that should be allowed to register with middleware.")
    memory: set[str] | None = Field(
        default=None, description="Additional Memory functions that should be allowed to register with middleware.")
    object_stores: set[str] | None = Field(
        default=None,
        description="Additional Object Store functions that should be allowed to register with middleware.")
    authentication: set[str] | None = Field(
        default=None,
        description="Additional Authentication functions that should be allowed to register with middleware.")

    @model_validator(mode='after')
    def merge_with_defaults(self):
        """Merge user-provided values with defaults from COMPONENT_FUNCTION_ALLOWLISTS."""
        from nat.middleware.utils.workflow_inventory import COMPONENT_FUNCTION_ALLOWLISTS

        def merge(component_group: ComponentGroup, user_set: set[str] | None) -> set[str]:
            defaults = COMPONENT_FUNCTION_ALLOWLISTS[component_group]
            if user_set is None:
                return defaults.copy()
            return defaults | user_set

        self.llms = merge(ComponentGroup.LLMS, self.llms)
        self.embedders = merge(ComponentGroup.EMBEDDERS, self.embedders)
        self.retrievers = merge(ComponentGroup.RETRIEVERS, self.retrievers)
        self.memory = merge(ComponentGroup.MEMORY, self.memory)
        self.object_stores = merge(ComponentGroup.OBJECT_STORES, self.object_stores)
        self.authentication = merge(ComponentGroup.AUTHENTICATION, self.authentication)

        return self


class DynamicMiddlewareConfig(FunctionMiddlewareBaseConfig, name="dynamic_middleware"):
    """Configuration for dynamic middleware.

    Controls which components and functions to intercept, and which policies to apply.
    Supports explicit component references and auto-discovery flags.
    """

    # === Component References ===

    llms: list[LLMRef] | None = Field(default=None, description="LLMs to intercept")

    embedders: list[EmbedderRef] | None = Field(default=None, description="Embedders component functions to intercept")

    retrievers: list[RetrieverRef] | None = Field(default=None,
                                                  description="Retrievers component functions to intercept")

    memory: list[MemoryRef] | None = Field(default=None, description="Memory component functions to intercept")

    object_stores: list[ObjectStoreRef] | None = Field(default=None,
                                                       description="Object stores component functions to intercept")

    auth_providers: list[AuthenticationRef] | None = Field(
        default=None, description="Authentication providers component functions to intercept")

    # === Component and Function Auto-Discovery Flags ===

    register_llms: bool | None = Field(default=False,
                                       description="Auto-discover and register all LLMs component functions")

    register_embedders: bool | None = Field(default=False,
                                            description="Auto-discover and register all embedders component functions")

    register_retrievers: bool | None = Field(
        default=False, description="Auto-discover and register all retrievers component functions")

    register_memory: bool | None = Field(
        default=False, description="Auto-discover and register all memory providers component functions")

    register_object_stores: bool | None = Field(
        default=False, description="Auto-discover and register all object stores component functions")

    register_auth_providers: bool | None = Field(
        default=False, description="Auto-discover and register all authentication providers component functions")

    register_workflow_functions: bool | None = Field(default=False,
                                                     description="Auto-discover and register all workflow functions")

    # === Enable/Disable ===

    enabled: bool = Field(default=True, description="Whether this middleware is active")

    # === Component Function Allowlists ===

    allowed_component_functions: AllowedComponentFunctions | None = Field(
        default=None,
        description="Functions allowed for auto-registration. Omit to use defaults, provide to extend them")
