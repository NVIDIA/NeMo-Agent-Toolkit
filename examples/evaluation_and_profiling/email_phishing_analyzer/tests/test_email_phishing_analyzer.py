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

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_run_full_workflow():
    from nat.runtime.loader import load_config
    from nat_email_phishing_analyzer.register import EmailPhishingAnalyzerConfig

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


@pytest.mark.skip(reason="This test gets rate limited potentially issue #842 and does not complete")
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "require_nest_asyncio")
async def test_optimize_full_workflow(capsys):
    from nat.data_models.config import Config
    from nat.data_models.optimizer import OptimizerRunConfig
    from nat.profiler.parameter_optimization.optimizer_runtime import optimize_config
    from nat_email_phishing_analyzer.register import EmailPhishingAnalyzerConfig

    config_file: Path = locate_example_config(EmailPhishingAnalyzerConfig, "config_optimizer.yml")
    config = OptimizerRunConfig(config_file=config_file,
                                dataset=None,
                                override=(('eval.general.max_concurrency', '1'), ('optimizer.numeric.n_trials', '1')))
    optimized_config = await optimize_config(config)
    assert isinstance(optimized_config, Config)
    captured_output = capsys.readouterr()

    assert "All optimization phases complete" in captured_output.out
