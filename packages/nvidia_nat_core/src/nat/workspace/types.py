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

import typing
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

from nat.data_models.workspace import ActionContext
from nat.data_models.workspace import ActionResult
from nat.data_models.workspace import WorkspaceBaseConfig


@dataclass
class TypeSchema:
    type: str
    description: str


@dataclass
class WorkspaceActionSchema:
    name: str
    description: str
    parameters: list[TypeSchema]
    result: TypeSchema


@dataclass
class WorkspaceSkillSchema:
    name: str
    description: str


class WorkspaceBase(ABC):

    @abstractmethod
    async def get_actions(self) -> list[WorkspaceActionSchema]:
        pass

    @abstractmethod
    async def get_skills(self) -> list[WorkspaceSkillSchema]:
        pass

    @abstractmethod
    async def create_skill(self, skill_name: str, skill_description: str) -> ActionResult:
        pass

    @abstractmethod
    async def execute_action(
        self,
        action_name: str,
        args: dict[str, typing.Any],
        model_metadata: dict[str, typing.Any] | None = None,
    ) -> ActionResult:
        pass

    @abstractmethod
    async def upload_file(self, file_path: Path, destination_path: Path) -> ActionResult:
        pass

    @abstractmethod
    async def delete_file(self, file_path: Path) -> ActionResult:
        pass

    @abstractmethod
    async def download_file(self, file_path: Path) -> ActionResult:
        pass

    @abstractmethod
    async def upload_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        pass

    @abstractmethod
    async def download_directory(self, directory_path: Path, destination_path: Path) -> ActionResult:
        pass

    @abstractmethod
    async def delete_directory(self, directory_path: Path) -> ActionResult:
        pass


ConfigT = typing.TypeVar("ConfigT", bound=WorkspaceBaseConfig)
InstanceT = typing.TypeVar("InstanceT", bound=WorkspaceBase)


class WorkspaceManagerBase(ABC, typing.Generic[ConfigT, InstanceT]):

    def __init__(self, config: ConfigT):
        self.config = config

    @abstractmethod
    async def __aenter__(self) -> InstanceT:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        pass


class WorkspaceAction(ABC):
    """Base interface for workspace actions."""

    name: str
    description: str
    parameters: list[TypeSchema]
    result: TypeSchema

    _context: ActionContext | None = None

    def set_context(self, context: ActionContext) -> None:
        """Bind the action to a session context."""
        self._context = context

    @classmethod
    def schema_for_registry(cls) -> WorkspaceActionSchema:
        """Return the schema using class attributes."""
        return WorkspaceActionSchema(
            name=cls.name,
            description=cls.description,
            parameters=cls.parameters,
            result=cls.result,
        )

    async def __aenter__(self) -> "WorkspaceAction":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @property
    def schema(self) -> WorkspaceActionSchema:
        """Return the action schema for API responses."""
        return type(self).schema_for_registry()

    @abstractmethod
    async def execute(self, context: ActionContext, args: dict[str, typing.Any]) -> typing.Any:
        """Execute the action with the provided args."""
        raise NotImplementedError
