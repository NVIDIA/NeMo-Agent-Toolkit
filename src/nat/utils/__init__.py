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

import typing
from pathlib import Path

if typing.TYPE_CHECKING:

    from nat.data_models.config import Config

    from .type_utils import StrPath

_T = typing.TypeVar("_T")


async def run_workflow(*,
                       config: "Config | None" = None,
                       config_file: "StrPath | None" = None,
                       prompt: str,
                       to_type: type[_T] = str,
                       session_kwargs: dict[str, typing.Any] | None = None) -> _T:
    """
    Wrapper to run a workflow given either a config or a config file path and a prompt, returning the result in the
    type specified by the `to_type`.

    Parameters
    ----------
    config : Config | None
        The configuration object to use for the workflow. If None, config_file must be provided.
    config_file : StrPath | None
        The path to the configuration file. If None, config must be provided. Can be either a str or a Path object.
    prompt : str
        The prompt to run the workflow with.
    to_type : type[_T]
        The type to convert the result to. Default is str.

    Returns
    -------
    _T
        The result of the workflow converted to the specified type.
    """
    from nat.builder.workflow_builder import WorkflowBuilder
    from nat.runtime.loader import load_config
    from nat.runtime.session import SessionManager

    if config is not None and config_file is not None:
        raise ValueError("Only one of config or config_file should be provided")

    if config is None:
        if config_file is None:
            raise ValueError("Either config_file or config must be provided")

        if not Path(config_file).exists():
            raise ValueError(f"Config file {config_file} does not exist")

        config = load_config(config_file)

    session_kwargs = session_kwargs or {}

    async with WorkflowBuilder.from_config(config=config) as workflow_builder:
        session_manager = SessionManager(await workflow_builder.build())
        async with session_manager.session(**session_kwargs) as session:
            async with session.run(prompt) as runner:
                return await runner.result(to_type=to_type)
