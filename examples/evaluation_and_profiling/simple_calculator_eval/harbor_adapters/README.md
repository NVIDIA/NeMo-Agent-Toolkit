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
The adapters convert example data files into Harbor task directories that can be
used with `harbor run`.

Use this README when you need to generate or inspect Harbor task directories.
For the actual `harbor run` commands, use the simple calculator example
cookbook linked at the end.

## Adapters

- `simple_calculator`: base arithmetic adapter. Pass a source file whose
  questions match the basic arithmetic parser. The tunable-rag Harbor example
  uses this adapter with the simple calculator dataset from
  `examples/getting_started/simple_calculator/data/simple_calculator.json`.
- `simple_calculator_power_of_two`: power-of-two focused dataset.

Shared generation behavior lives in `common.py`. Each adapter only defines how
to load source rows and how to customize task-specific prompt, metadata, and
ground-truth fields.

## Generate Tasks

Run adapter scripts from the repository root.

Generate the simple calculator dataset used by the tunable-rag Harbor example:

```bash
NAT_HARBOR_TUNABLE_RAG_ADAPTER=examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/simple_calculator/run_adapter.py
NAT_HARBOR_TUNABLE_RAG_SOURCE=examples/getting_started/simple_calculator/data/simple_calculator.json
NAT_HARBOR_TUNABLE_RAG_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-tunable-rag

python "$NAT_HARBOR_TUNABLE_RAG_ADAPTER" \
  --source-file "$NAT_HARBOR_TUNABLE_RAG_SOURCE" \
  --output-dir "$NAT_HARBOR_TUNABLE_RAG_DATASET_DIR" \
  --overwrite
```

Generate the power-of-two dataset:

```bash
NAT_HARBOR_POWER_OF_TWO_ADAPTER=examples/evaluation_and_profiling/simple_calculator_eval/harbor_adapters/simple_calculator_power_of_two/run_adapter.py
NAT_HARBOR_POWER_OF_TWO_DATASET_DIR=.tmp/harbor/datasets/simple-calculator-power-of-two

python "$NAT_HARBOR_POWER_OF_TWO_ADAPTER" \
  --output-dir "$NAT_HARBOR_POWER_OF_TWO_DATASET_DIR" \
  --overwrite
```

Useful flags:

- `--source-file`: read a different source JSON file.
- `--output-dir`: write generated Harbor tasks to a different directory.
- `--ids`: generate only specific source IDs.
- `--limit`: generate only the first N source rows after ID selection.
- `--overwrite`: replace existing generated task directories.

## Output

By default, adapter scripts write tasks under:

`external/harbor/datasets/<adapter-name>`

For local development, prefer writing under `.tmp/harbor/datasets/...` so the
generated tasks stay out of the source tree.

## Task Directory Structure

Each generated task is self-contained and follows the Harbor task layout:

```text
simple-calculator-power-of-two-1/
  environment/
    Dockerfile
  instruction.md
  solution/
    solve.sh
  task.toml
  tests/
    evaluate.py
    ground_truth.json
    test.sh
```

Important files:

- `instruction.md`: prompt Harbor gives the agent.
- `task.toml`: Harbor task metadata, timeouts, and environment settings.
- `solve.sh`: oracle solution used for compatibility with Harbor task
  conventions.
- `ground_truth.json`: expected numeric answer and verifier metadata.
- `test.sh`: verifier entrypoint invoked by Harbor script verification.
- `evaluate.py`: strict numeric verifier shared by these tasks.

## Related docs

- [Package README](../../../../packages/nvidia_nat_harbor/README.md)
- [Simple calculator example cookbook](../harbor-eval-readme.md)
