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
"""A handler in `general.telemetry.logging` gets records at its own level, also below the CLI's `--log-level`."""

import io
import logging
from pathlib import Path

import pytest

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.builder.workflow_builder import WorkflowBuilder
from nat.cli.register_workflow import register_function
from nat.data_models.config import Config
from nat.data_models.function import FunctionBaseConfig
from nat.runtime.session import SessionManager

WORKFLOW_LOGGER = "nat.test_logging_config_level"
USER_PACKAGE_LOGGER = "my_workflow_package"  # a workflow package outside `nat`


class LogsAtEveryLevelConfig(FunctionBaseConfig, name="logging_config_level_test_workflow"):
    pass


@pytest.fixture(name="register_workflow", scope="module", autouse=True)
def register_workflow_fixture():

    @register_function(config_type=LogsAtEveryLevelConfig)
    async def _workflow(config, builder: Builder):

        async def _inner(text: str) -> str:
            logging.getLogger(WORKFLOW_LOGGER).debug("debug-record")
            logging.getLogger(WORKFLOW_LOGGER).info("info-record")
            logging.getLogger(USER_PACKAGE_LOGGER).debug("user-package-debug-record")
            return text

        yield FunctionInfo.from_fn(_inner, description="logs at DEBUG and INFO")


@pytest.fixture(name="cli_console")
def cli_console_fixture():
    """The logging `nat` sets up before a command runs: a root console handler, root and `nat` at INFO."""
    root_logger = logging.getLogger()
    nat_logger = logging.getLogger("nat")
    saved = (root_logger.level, nat_logger.level, root_logger.handlers[:])
    console = io.StringIO()
    handler = logging.StreamHandler(console)
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    nat_logger.setLevel(logging.INFO)
    try:
        yield console
    finally:
        root_logger.setLevel(saved[0])
        nat_logger.setLevel(saved[1])
        root_logger.handlers = saved[2]


async def _run(log_file: Path, *, with_console: bool, file_level: str = "DEBUG"):
    logging_config = {"file": {"_type": "file", "path": str(log_file), "level": file_level}}
    if with_console:
        logging_config["console"] = {"_type": "console", "level": "WARN"}
    config = Config.model_validate({
        "general": {
            "telemetry": {
                "logging": logging_config
            }
        },
        "workflow": {
            "_type": "logging_config_level_test_workflow"
        },
    })
    async with WorkflowBuilder.from_config(config=config) as builder:
        session_manager = await SessionManager.create(config=config, shared_builder=builder)
        try:
            async with session_manager.session() as session:
                async with session.run("hi") as runner:
                    assert await runner.result(to_type=str) == "hi"
        finally:
            await session_manager.shutdown()


@pytest.mark.parametrize("with_console", [True, False], ids=["file_and_console", "file_only"])
async def test_a_debug_file_handler_gets_debug_records(tmp_path: Path, cli_console: io.StringIO, with_console: bool):
    log_file = tmp_path / "workflow.log"
    await _run(log_file, with_console=with_console)

    content = log_file.read_text(encoding="utf-8")
    assert "debug-record" in content
    assert "info-record" in content
    assert "user-package-debug-record" in content


async def test_the_cli_console_keeps_its_level(tmp_path: Path, cli_console: io.StringIO):
    """A DEBUG file doesn't turn the console the CLI set up at INFO into a DEBUG console, or lower other handlers."""
    warning_output = io.StringIO()
    warning_handler = logging.StreamHandler(warning_output)
    warning_handler.setLevel(logging.WARNING)
    logging.getLogger().addHandler(warning_handler)
    try:
        await _run(tmp_path / "workflow.log", with_console=False)
    finally:
        logging.getLogger().removeHandler(warning_handler)

    console_output = cli_console.getvalue()
    assert "info-record" in console_output
    assert "debug-record" not in console_output
    assert warning_output.getvalue() == ""


@pytest.mark.parametrize("with_console", [True, False], ids=["file_and_console", "file_only"])
async def test_logger_levels_are_restored_when_the_builder_exits(tmp_path: Path,
                                                                 cli_console: io.StringIO,
                                                                 with_console: bool):
    root_logger = logging.getLogger()
    handlers_before = root_logger.handlers[:]  # the CLI console, plus pytest's capture handler
    await _run(tmp_path / "workflow.log", with_console=with_console)

    assert root_logger.level == logging.INFO
    assert logging.getLogger("nat").level == logging.INFO
    assert set(root_logger.handlers) == set(handlers_before)
    assert all(handler.level == logging.NOTSET for handler in handlers_before)


async def test_a_handler_above_the_cli_level_changes_no_levels(tmp_path: Path, cli_console: io.StringIO):
    """With --log-level DEBUG and a WARNING file, the root logger stays at DEBUG and other handlers keep theirs."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("nat").setLevel(logging.DEBUG)
    warning_handler = logging.StreamHandler(io.StringIO())
    warning_handler.setLevel(logging.WARNING)
    root_logger.addHandler(warning_handler)
    try:
        await _run(tmp_path / "workflow.log", with_console=False, file_level="WARNING")
        # a logger without its own level follows the root logger, which must not have been raised to WARNING
        assert "user-package-debug-record" in cli_console.getvalue()
    finally:
        root_logger.removeHandler(warning_handler)

    assert root_logger.level == logging.DEBUG
    assert warning_handler.level == logging.WARNING
