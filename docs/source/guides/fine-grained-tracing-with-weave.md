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

# Observing a Workflow with W&B Weave

This guide provides a step-by-step process to enable observability in an AIQ Toolkit workflow using Weights and Biases (W&B) Weave for tracing using just a few lines of code in your workflow configuration file.

![Weave Tracing Dashboard](../_static/weave_tracing.png)

### Step 1: Install the Weave plugin

To install the Weave plugin, run the following:

```bash
uv pip install -e '.[weave]'
```

### Step 2: Install the Workflow

Pick an example from the list of available workflows. In this guide, we will be using the `simple_calculator` example.

```bash
uv pip install -e examples/simple_calculator
```

### Step 3: Modify Workflow Configuration

Update your workflow configuration file to include the weave telemetry settings. For example, `examples/simple_calculator/configs/config-weave.yml` has the following weave settings:

```bash
general:
  use_uvloop: true
  telemetry:
    tracing:
      weave:
        _type: weave
        project: "aiqtoolkit-demo"
```

This setup enables logging trace data to W&B weave. The weave integration requires one parameter and one optional parameter:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `project` | The name of your W&B Weave project | `"aiqtoolkit-demo"` |
| `entity` (optional) | Your W&B username or team name | `"your-wandb-username-or-teamname"` |

### Step 4: Run Your Workflow
From the root directory of the AIQ Toolkit library, execute your workflow as shown below:

```bash
aiq run --config_file examples/simple_calculator/configs/config.yml --input "Is the product of 2 * 4 greater than the current hour of the day?"
```

If it is your first time running the workflow, you will be prompted to login to W&B Weave.

### Step 5: View Traces Data in Weave Dashboard

As the workflow runs, you will find a Weave URL (starting with a 🍩 emoji). Click on the URL to access your logged trace timeline.

Note how the integration captures not only the `aiq` intermediate steps but also the underlying framework. This is because [Weave has integrations](https://weave-docs.wandb.ai/guides/integrations/) with many of your favorite frameworks.

## Resources

- Learn more about tracing [here](https://weave-docs.wandb.ai/guides/tracking/tracing).
- Learn more about how to navigate the logged traces [here](https://weave-docs.wandb.ai/guides/tracking/trace-tree).
