# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from aiq.data_models.authentication import AuthProviderBaseConfig
from aiq.data_models.embedder import EmbedderBaseConfig
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.llm import LLMBaseConfig
from aiq.data_models.memory import MemoryBaseConfig
from aiq.data_models.object_store import ObjectStoreBaseConfig
from aiq.data_models.registry_handler import RegistryHandlerBaseConfig


class WorkflowTestConfig(FunctionBaseConfig, name="test_workflow"):
    llm_name: str
    functions: list[str]
    prompt: str


class FunctionTestConfig(FunctionBaseConfig, name="test_function"):
    pass


class ToolDocstringTestConfig(FunctionBaseConfig, name="test_tool_docstring"):
    pass


class ToolNoDescriptionTestConfig(FunctionBaseConfig, name="test_tool_no_description"):
    pass


class LLMProviderTestConfig(LLMBaseConfig, name="test_llm"):
    pass


class EmbedderProviderTestConfig(EmbedderBaseConfig, name="test_embedding"):
    pass


class MemoryTestConfig(MemoryBaseConfig, name="test_memory"):
    pass


class ObjectStoreTestConfig(ObjectStoreBaseConfig, name="test_object_store"):
    pass


class RegistryHandlerTestConfig(RegistryHandlerBaseConfig, name="test_registry_handler"):
    pass


class AuthenticationProviderTestConfig(AuthProviderBaseConfig, name="test_authentication"):
    pass
