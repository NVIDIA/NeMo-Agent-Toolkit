<!--
    SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
-->

# Running CI Locally

The `ci/scripts/run_ci_local.sh` attempts to mirror the behavior of the GitHub Actions CI pipeline, by running the same CI scripts within the same Docker container that is used in the CI pipeline with the same environment variables needed.

By default the script will perform a `git clone` and checkout the latest commit. This requires the latest commit to be pushed. Alternately setting the environment variable `USE_HOST_GIT=1` the host's repo will be mounted inside the CI container, avoiding the need to commit/push changes. This option requires all Git LFS files to be checked out first.

> Note: We don't set all of the same environment variables that GitHub Actions would set, just the ones needed by our own CI scripts.


## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- AgentIQ source repository cloned locally with both the `origin` and `upstream` remotes set up. Refer to [Creating the Environment](./contributing.md#creating-the-environment) for more details.

## Usage
Typical usage is as follows:
```bash
./ci/scripts/run_ci_local.sh <CI stage>
```

For example, to run the `checks` stage:
```bash
./ci/scripts/run_ci_local.sh checks
```

To run all CI stages, you can use:
```bash
./ci/scripts/run_ci_local.sh all
```

## Debugging CI

To debug a CI issue, you can use the `bash` pseudo-stage. This will perform a git clone & checkout, and then drop you in a bash shell with all of the CI variables set.
```bash
./ci/scripts/run_ci_local.sh bash
```

From this point you can manually copy/paste the commands which would normally be run by the CI scripts one command at a time. The GitHub Actions CI scripts for AgentIQ are located in the `ci/scripts/github` directory, these scripts are GitHub Actions specific wrappers for scripts located in the `ci/scripts` directory.

## CI Artifacts and Cache

| Name | Description | Location |
|--|--|--|
| Artifacts | Test results, wheels, and documentation | `.tmp/local_ci_tmp/local_ci_workspace` |
| Cache | `uv` and `pre-commit` package caches | `.tmp/local_ci_tmp/cache` |
| Virtual Environment | Python virtual environment | `.tmp/local_ci_tmp/local_ci_workspace/.venv` |
| Bootstrap Script | The script used to bootstrap the CI environment within the CI container | `.tmp/local_ci_tmp/bootstrap_local_ci.sh` |

> Note: In some situations it may be necessary to delete the `.tmp/local_ci_tmp` directory to clear out old artifacts and caches. This is especially true if you are switching between branches or if you are running into issues with the CI pipeline.
