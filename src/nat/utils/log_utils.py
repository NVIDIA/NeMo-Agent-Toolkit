# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

LOG_FORMAT = "%(asctime)s - %(levelname)-8s - %(name)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class LogFilter(logging.Filter):
    """
    This class is used to filter log records based on a defined set of criteria.
    """

    def __init__(self, filter_criteria: list[str]):
        self._filter_criteria = filter_criteria
        super().__init__()

    def filter(self, record: logging.LogRecord):
        """
        Evaluates whether a log record should be emitted based on the message content.

        Returns:
            False if the message content contains any of the filter criteria, True otherwise.
        """
        if any(match in record.getMessage() for match in self._filter_criteria):
            return False
        return True


def setup_logging(log_level: int):
    """Configure logging with the specified level"""
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
