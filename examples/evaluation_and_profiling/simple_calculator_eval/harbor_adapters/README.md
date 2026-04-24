<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
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

