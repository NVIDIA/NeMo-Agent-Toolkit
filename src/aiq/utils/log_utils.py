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
