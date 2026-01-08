<!--
 SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Building Documentation

## Prerequisites
If you don't already have a uv environment setup, refer to the [Get Started](./source/get-started/installation.md) guide.

## Install Documentation Dependencies
```bash
uv sync --all-groups --all-extras
```

## Build Documentation
<!-- path-check-skip-begin -->
```bash
make -C docs

# preview with local server (open http://localhost:8000 in your browser)
python -m http.server --directory docs/build/html 8000

```
<!-- path-check-skip-next-line -->
Outputs to `docs/build/docs/html`

### Optional Quick Build Command

A full documentation build can take several minutes. The time consuming steps are building the Python API and performing the link check. 

To skip both of these steps, you can use the following command:
```bash
NAT_DISABLE_API_BUILD=1 make -C docs html
```

To run the link check separately, use:
```bash
make -C docs linkcheck
```

**Note**: When viewing documentation locally, the version switcher in the navigation bar will redirect to the production documentation site (`https://docs.nvidia.com/nemo/agent-toolkit/`) when selecting a different version. This is expected behavior, as the version switcher uses absolute URLs to ensure proper page path preservation in production.

## Contributing
Refer to the [Contributing to NeMo Agent toolkit](./source/resources/contributing/index.md) guide.

When you create your pull request, CI will perform a documentation build as part of the pipeline. If successful, the documentation will be available for download as an artifact.
