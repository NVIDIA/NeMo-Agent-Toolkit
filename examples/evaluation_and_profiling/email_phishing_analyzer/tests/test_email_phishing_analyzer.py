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
from pathlib import Path

import pytest

from nat.test.utils import locate_example_config
from nat.test.utils import run_workflow
from nat_email_phishing_analyzer.register import EmailPhishingAnalyzerConfig

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow(milvus_uri: str) -> None:
    from pydantic import HttpUrl

    from nat.runtime.loader import load_config

    config_file: Path = locate_example_config(EmailPhishingAnalyzerConfig)
    config = load_config(config_file)

    # Unfortunately the workflow itself returns inconsistent results
    await run_workflow(
        config=config,
        question=(
            "Dear [Customer], Thank you for your purchase on [Date]. We have processed a refund of $[Amount] to your "
            "account. Please provide your account and routing numbers so we can complete the transaction. Thank you, "
            "[Your Company]"),
        expected_answer="likely")
