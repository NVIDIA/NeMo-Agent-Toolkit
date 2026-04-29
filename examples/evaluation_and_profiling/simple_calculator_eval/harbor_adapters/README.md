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

# Harbor Adapters for Simple Calculator Eval

This directory hosts Harbor task adapters owned by the simple calculator example.

## Adapters

- `simple_calculator`: base arithmetic dataset
- `simple_calculator_nested`: nested-flow arithmetic dataset
- `simple_calculator_power_of_two`: power-of-two focused dataset

## Output

By default, adapter scripts write tasks under:

`external/harbor/datasets/<adapter-name>`

Each script supports overriding source and output paths via CLI flags.

## Future direction

The current adapters are concrete reference implementations. A follow-up should
extract the repeated task-generation flow into shared `nat_harbor.adapters`
helpers so future NeMo Agent Toolkit examples only define source-row loading, instruction
rendering, and ground-truth mapping.
