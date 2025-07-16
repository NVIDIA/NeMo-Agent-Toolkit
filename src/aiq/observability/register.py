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

import logging

from pydantic import Field

from aiq.builder.builder import Builder
from aiq.cli.register_workflow import register_logging_method
from aiq.cli.register_workflow import register_telemetry_exporter
from aiq.data_models.logging import LoggingBaseConfig
from aiq.data_models.telemetry_exporter import TelemetryExporterBaseConfig

logger = logging.getLogger(__name__)


class FileTelemetryExporter(TelemetryExporterBaseConfig, name="file"):
    """A logger to write runtime logs to the console."""

    filepath: str = Field(description="The file path to log traces.")
    project: str = Field(description="Name to affilite with this application.")


@register_telemetry_exporter(config_type=FileTelemetryExporter)
async def file_telemetry_exporter(config: FileTelemetryExporter, builder: Builder):  # pylint: disable=W0613
    """
    Build and return a StreamHandler for console-based logging.
    """

    from aiq.observability.exporter.file_exporter import FileExporter

    yield FileExporter(filepath=config.filepath, project=config.project)


class ConsoleLoggingMethod(LoggingBaseConfig, name="console"):
    """A logger to write runtime logs to the console."""

    level: str = Field(description="The logging level of console logger.")


@register_logging_method(config_type=ConsoleLoggingMethod)
async def console_logging_method(config: ConsoleLoggingMethod, builder: Builder):  # pylint: disable=W0613
    """
    Build and return a StreamHandler for console-based logging.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    yield handler


class FileLoggingMethod(LoggingBaseConfig, name="file"):
    """A logger to write runtime logs to a file."""

    path: str = Field(description="The file path to save the logging output.")
    level: str = Field(description="The logging level of file logger.")


@register_logging_method(config_type=FileLoggingMethod)
async def file_logging_method(config: FileLoggingMethod, builder: Builder):  # pylint: disable=W0613
    """
    Build and return a FileHandler for file-based logging.
    """
    level = getattr(logging, config.level.upper(), logging.INFO)
    handler = logging.FileHandler(filename=config.path, mode="a", encoding="utf-8")
    handler.setLevel(level)
    yield handler
