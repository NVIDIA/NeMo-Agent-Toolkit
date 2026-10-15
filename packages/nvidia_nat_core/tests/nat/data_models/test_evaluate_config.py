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

from pathlib import Path

from nat.data_models.evaluate_config import EvalGeneralConfig


def test_output_section_without_dir_uses_output_dir():
    """`output_dir` plus an `output` section with no `dir`, as in the documented S3 upload snippet.

    Results are written to `output_dir`, while cleanup, custom scripts and the S3 upload read `output.dir`,
    so both must name the same directory.
    """
    general = EvalGeneralConfig.model_validate({
        "output_dir": "./.tmp/my_eval/",
        "output": {
            "remote_dir": "output", "cleanup": True
        },
    })

    assert general.output_dir == Path("./.tmp/my_eval/")
    assert general.output.dir == general.output_dir


def test_output_section_dir_still_overrides_output_dir():
    general = EvalGeneralConfig.model_validate({
        "output_dir": "./.tmp/ignored/",
        "output": {
            "dir": "./.tmp/chosen/"
        },
    })

    assert general.output_dir == general.output.dir == Path("./.tmp/chosen/")
