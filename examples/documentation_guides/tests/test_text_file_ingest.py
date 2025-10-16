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
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

from nat.test.utils import locate_example_config
from nat.test.utils import run_workflow

logger = logging.getLogger(__name__)


@pytest.fixture(name="text_file_ingest_dir", scope="session")
def text_file_ingest_dir_fixture(workflows_dir: Path) -> Path:
    text_file_ingest = workflows_dir / "text_file_ingest"
    assert text_file_ingest.exists(), f"Could not find text_file_ingest example at {text_file_ingest}"
    return text_file_ingest


@pytest.fixture(name="src_dir", scope="session", autouse=True)
def src_dir_fixture(text_file_ingest_dir: Path) -> Path:
    src_dir = text_file_ingest_dir / "src"
    assert src_dir.exists(), f"Could not find text_file_ingest src at {src_dir}"

    return src_dir


@pytest.fixture(name="add_src_dir_to_path", scope="session")
def add_src_dir_to_path_fixture(src_dir: Path) -> Generator[str]:
    # Since this is a documentation guide, it is not installed by default, so we need to manually append it to the path
    abs_src_dir = str(src_dir.absolute())
    if abs_src_dir not in sys.path:
        added = True
        sys.path.append(abs_src_dir)
    else:
        added = False

    yield abs_src_dir

    if added:
        sys.path.remove(abs_src_dir)


@pytest.mark.integration
@pytest.mark.usefixtures("nvidia_api_key", "add_src_dir_to_path")
async def test_text_file_ingest_full_workflow():
    from text_file_ingest.text_file_ingest_function import TextFileIngestFunctionConfig
    config_file = locate_example_config(TextFileIngestFunctionConfig)
    await run_workflow(config_file=config_file,
                       question="What does DOCA GPUNetIO do to remove the CPU from the critical path?",
                       expected_answer="GPUDirect")
