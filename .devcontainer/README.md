<!--
SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# AIQ Toolkit Devcontainer

The AIQ Toolkit devcontainer is provided as a quick-to-set-up development and exploration environment for use with [Visual Studio Code](https://code.visualstudio.com) (Code). The devcontainer is a lightweight container which mounts-in a Conda environment with cached packages, alleviating long Conda download times on subsequent launches. It provides a simple framework for adding developer-centric [scripts](#development-scripts), and incorporates some helpful Code plugins, such as clangd and CMake support.

More information about devcontainers can be found at [`containers.dev`](https://containers.dev/).

## Get Started

To get started, simply open the AIQ Toolkit repository root folder within Code. A window should appear at the bottom-right corner of the editor asking if you would like to reopen the workspace inside of the dev container. After clicking the confirmation dialog, the container will first build, then launch, then remote-attach.

If the window does not appear, or you would like to rebuild the container, click ctrl-shift-p and search for `Dev Containers: Rebuild and Reopen in Container`. Hit enter, and the container will first build, then launch, then remote-attach.

Once connected to the devcontainer within code, the `setup-aiq-env` script will begin to run and solve a AIQ Toolkit Conda environment (this Conda environment is local to the AIQ Toolkit repository and dev container and will not override any host environments). You should see the script executing in one of Code's integrated terminal. Once the script has completed, we're ready to start development or exploration of AIQ Toolkit. By default, each _new_ integrated terminal will automatically Conda activate the AIQ Toolkit environment.

## Development Scripts
Several convenient scripts are available in the devcontainer's `PATH` (`.devcontainer/bin`) for starting, stopping, and interacting with Triton and Kafka. More scripts can be added as needed.
