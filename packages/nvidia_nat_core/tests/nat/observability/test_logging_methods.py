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

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import nullcontext
from pathlib import Path

import pytest
import yaml

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.cli.type_registry import TypeRegistry
from nat.data_models.function import FunctionBaseConfig
from nat.runtime.loader import load_workflow
from nat.runtime.session import SessionManager


@pytest.fixture(name="logging_workflow")
def logging_workflow_fixture(tmp_path: Path, registry: TypeRegistry) -> Path:

    class LoggingWorkflowConfig(FunctionBaseConfig, name="file_logging_lifetime_test"):
        pass

    @register_function(config_type=LoggingWorkflowConfig)
    async def build_workflow(config: LoggingWorkflowConfig, builder: Builder) -> AsyncGenerator[FunctionInfo, None]:

        async def run(message: str) -> str:
            logging.getLogger(__name__).warning("workflow-record: %s", message)
            if message == "fail":
                raise ValueError("workflow failure")
            return message

        yield FunctionInfo.from_fn(run, description="Exercise managed file logging.")

    config_file: Path = tmp_path / "workflow.yml"
    config_file.write_text(
        yaml.safe_dump({
            "general": {
                "telemetry": {
                    "logging": {
                        "file_log": {
                            "_type": "file",
                            "path": str(tmp_path / "shared.log"),
                            "level": "WARNING",
                        }
                    }
                }
            },
            "workflow": {
                "_type": "file_logging_lifetime_test"
            },
        }),
        encoding="utf-8",
    )
    return config_file


async def run_workflow(manager: SessionManager, message: str) -> str:
    async with manager.session() as session:
        async with session.run(message=message) as runner:
            return await runner.result(to_type=str)


@pytest.mark.parametrize("fail", [False, True])
async def test_file_logging_closes_only_its_owned_handler(logging_workflow: Path, tmp_path: Path, fail: bool) -> None:
    logfile: Path = tmp_path / "shared.log"
    host_handler = logging.FileHandler(filename=logfile, encoding="utf-8")
    host_handler.setLevel(logging.WARNING)
    root_logger: logging.Logger = logging.getLogger()
    root_logger.addHandler(host_handler)
    owned_handlers: list[logging.FileHandler] = []
    try:
        async with load_workflow(config_file=logging_workflow, max_concurrency=1) as outer:
            outer_handler: logging.Handler = outer.shared_builder._logging_handlers["file_log"]
            assert isinstance(outer_handler, logging.FileHandler)
            owned_handlers.append(outer_handler)
            with pytest.raises(ValueError, match="workflow failure") if fail else nullcontext():
                async with load_workflow(config_file=logging_workflow, max_concurrency=1) as inner:
                    inner_handler: logging.Handler = inner.shared_builder._logging_handlers["file_log"]
                    assert isinstance(inner_handler, logging.FileHandler)
                    owned_handlers.append(inner_handler)
                    assert host_handler.stream is not None
                    assert outer_handler.stream is not None
                    assert inner_handler.stream is not None
                    assert len({
                        host_handler.stream.fileno(),
                        outer_handler.stream.fileno(),
                        inner_handler.stream.fileno(),
                    }) == 3
                    result: str = await run_workflow(manager=inner, message="fail" if fail else "inner")
                    assert result == "inner"
            inner_closed: bool = inner_handler.stream is None
            assert inner_handler not in root_logger.handlers
            assert outer_handler in root_logger.handlers
            assert outer_handler.stream is not None and not outer_handler.stream.closed
            assert await run_workflow(manager=outer, message="outer-survives") == "outer-survives"
        outer_closed: bool = outer_handler.stream is None
        assert outer_handler not in root_logger.handlers
        assert host_handler in root_logger.handlers
        assert host_handler.stream is not None and not host_handler.stream.closed
        root_logger.warning("host-survives")
        content: str = logfile.read_text(encoding="utf-8")
        assert "workflow-record: outer-survives" in content
        assert "host-survives" in content
        print(json.dumps({
            "workflow_failed": fail,
            "inner_handler_closed": inner_closed,
            "outer_handler_closed": outer_closed,
            "surviving_workflow_logged": "workflow-record: outer-survives" in content,
            "host_handler_open": not host_handler.stream.closed,
        }),
              flush=True)
        assert inner_closed, "The inner workflow left its file handler open"
        assert outer_closed, "The outer workflow left its file handler open"
    finally:
        root_logger.removeHandler(host_handler)
        host_handler.close()
        for handler in owned_handlers:
            handler.close()
