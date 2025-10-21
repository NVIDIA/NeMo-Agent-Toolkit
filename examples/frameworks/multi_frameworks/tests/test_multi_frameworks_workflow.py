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

import pytest


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key")
async def test_full_workflow():
    from nat.test.utils import locate_example_config
    from nat.test.utils import run_workflow
    from nat_multi_frameworks.register import MultiFrameworksWorkflowConfig

    config_file = locate_example_config(MultiFrameworksWorkflowConfig)

    await run_workflow(config_file=config_file, question="tell me about this workflow", expected_answer="workflow")
