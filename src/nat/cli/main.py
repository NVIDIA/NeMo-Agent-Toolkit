# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


# The purpose of this function is to allow loading the current directory as a module. This allows relative imports and
# more specifically `..common` to function correctly
def run_cli():
    import os
    import sys

    # Suppress warnings from transformers
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if (parent_dir not in sys.path):
        sys.path.append(parent_dir)

    from nat.cli.entrypoint import cli

    cli(obj={}, auto_envvar_prefix='NAT', show_default=True, prog_name="nat")


def run_cli_aiq_compat():
    "Entrypoint for the `aiq` compatibility command"
    import warnings

    # Warn with a UserWarning since DeprecationWarnings are not shown by default
    warnings.warn(
        "The 'aiq' command is deprecated and will be removed in a future release. "
        "Please use the 'nat' command instead.",
        UserWarning,
        stacklevel=2)
    run_cli()


if __name__ == '__main__':
    run_cli()
