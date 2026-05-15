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

# ATIF trajectory scores (optional post-run)

OpenClaw does not produce an ATOF-derived second ATIF path like the OpenCode
NeMo-Flow smoke (`agent/nemo-flow-atof-atif/trajectory.json`). Post-run
`nat_harbor.smoke.score_atif_trajectories` is therefore most useful against
`agent/trajectory.json` alone unless you add a separate candidate ATIF.

The table below records **placeholders** for the same columns as
`opencode-nemoflow-smoke/atif-trajectory-scores.md` so tooling that expects this
file can still parse it; fill in after you run the scorer locally.

| Task | Reward | Deterministic | Native | NeMo-Flow | Delta | Category | Errors |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- |
| django__django-13741 | 1 | see `atif-tool-compare.md` | — | — | — | `not_applicable` | LLM scorer not run for fixture drop |
