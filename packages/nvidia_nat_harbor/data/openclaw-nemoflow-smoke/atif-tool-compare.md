<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Deterministic tool comparison (native vs plugin export)

Captured from:

```bash
python -m nat_harbor.smoke.compare_atif_tools \
  --native agent/trajectory.json \
  --candidate agent/nemo-flow-atif/nemo-flow-atif-019e2d16-f8a2-7fe2-80c6-5c4cf7efb697.json
```

Result:

```text
Classification: match (poorer)
Native tools (60): edit=6, exec=34, read=10, web_fetch=4, web_search=5, write=1
Candidate tools (0): (none)
```

The NeMo Flow OpenClaw plugin file is an observability bundle (session scope +
`observed_events`), not the same on-disk shape as an ATIF trajectory consumed
by `compare_atif_tools`. Treat this comparison as diagnostic only; the
checked-in evidence trajectory is `native-openclaw-trajectory.json`.
