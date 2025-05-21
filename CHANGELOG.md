<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Changelog
All notable changes to this project will be documented in this file.

## [1.1.0] - 2025-05-16
### Key Features
- Full MCP (Model Context Protocol) support
- Weave tracing
- Agno integration
- ReWOO Agent
- Alert Triage Agent Example

### What's Changed
* Have the examples README point to the absolute path by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/4
* Set initial version will be 1.0.0 by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/7
* Update `examples/simple_rag/README.md` to verify the installation of `lxml` by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/9
* Use a separate README for pypi by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/10
* Document the need to install from source to run examples by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/8
* Fixing broken links by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/14
* Cleanup readmes by @mdemoret-nv in https://github.com/NVIDIA/AIQToolkit/pull/15
* Pypi readme updates by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/16
* Final 1.0.0 cleanup by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/18
* Add subpackage readmes redirecting to the main package by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/20
* Update README.md by @gzitzlsb-nv in https://github.com/NVIDIA/AIQToolkit/pull/25
* Fix #27 Documentation fix by @atalhens in https://github.com/NVIDIA/AIQToolkit/pull/28
* Fix #29 - Simple_calculator example throws error - list index out of range when given subtraction by @atalhens in https://github.com/NVIDIA/AIQToolkit/pull/31
* Fix: #32 Recursion Issue by @atalhens in https://github.com/NVIDIA/AIQToolkit/pull/33
* "Sharing NVIDIA AgentIQ Components" docs typo fix by @avoroshilov in https://github.com/NVIDIA/AIQToolkit/pull/42
* First pass at setting up issue templates by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/6
* Provide a cleaner progress bar when running evaluators in parallel by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/38
* Setup GHA CI by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/46
* Switch UI submodule to https by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/53
* gitlab ci pipeline cleanup by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/54
* Allow str or None for retriever description by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/55
* Fix case where res['categories'] = None by @balvisio in https://github.com/NVIDIA/AIQToolkit/pull/22
* Misc CI improvements by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/56
* CI Documentation improvements by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/24
* Add missing `platformdirs` dependency by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/62
* Fix `aiq` command error when the parent directory of `AIQ_CONFIG_DIR` does not exist by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/63
* Fix broken image link in multi_frameworks documentation by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/61
* Updating doc string for AIQSessionManager class. by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/64
* Fix ragas evaluate unit tests by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/68
* Normalize Gannt Chart Timestamps in Profiler Nested Stack Analysis by @dnandakumar-nv in https://github.com/NVIDIA/AIQToolkit/pull/70
* Scripts for running CI locally by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/59
* Update types for `topic` and `description` attributes in  `AIQRetrieverConfig` to allow `None` by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/76
* Add support for customizing output and uploading it to remote storage (S3 bucket) by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/71
* Support ARM in CI by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/73
* Allow overriding configuration values not set in the YAML by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/85
* Fix bug where `--workers` flag was being ignored by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/88
* Adding Cors config for api server by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/89
* Update changelog for 1.1.0a1 alpha release by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/90
* Updated changelog with another bug fix by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/93
* Adjust how the base_sha is passed into the workflow by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/81
* Changes for evaluating remote workflows by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/57
* Fix a bug in our pytest plugin causing test coverage to be under-reported by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/105
* Docker container for AgentIQ by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/87
* Modify JSON serialization to handle non-serializable objects by @dnandakumar-nv in https://github.com/NVIDIA/AIQToolkit/pull/106
* Upload nightly builds and release builds to pypi by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/112
* Ensure the nightly builds have a unique alpha version number by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/115
* Ensure tags are fetched prior to determining the version by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/116
* Fix CI variable value by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/117
* Use setuptools_scm environment variables to set the version by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/118
* Only set the setuptools_scm variable when performing a nightly build by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/119
* Add a release PR template by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/123
* Add an async /evaluate endpoint to trigger evaluation jobs on a remote cluster by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/109
* Update /evaluate endpoint doc by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/126
* Add function tracking decorator and update IntermediateStep by @dnandakumar-nv in https://github.com/NVIDIA/AIQToolkit/pull/98
* Fix typo in aiq.profiler.decorators by @dnandakumar-nv in https://github.com/NVIDIA/AIQToolkit/pull/132
* Update the start command to use `validate_schema` by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/82
* Document using local/self-hosted models by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/101
* added Agno integration by @wenqiglantz in https://github.com/NVIDIA/AIQToolkit/pull/36
* MCP Front-End Implementation by @VictorYudin in https://github.com/NVIDIA/AIQToolkit/pull/133
* Make kwargs optional to the eval output customizer scripts by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/139
* Add an example that shows simple_calculator running with a MCP service. by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/131
* add `gitdiagram` to README by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/141
* Updating HITL reference guide to instruct users to toggle ws mode and… by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/142
* Add override option to the eval CLI command by @Hritik003 in https://github.com/NVIDIA/AIQToolkit/pull/129
* Implement ReWOO Agent by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/75
* Fix type hints and docstrings for `ModelTrainer` by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/107
* Delete workflow confirmation check in CLI - #114 by @atalhens in https://github.com/NVIDIA/AIQToolkit/pull/137
* Improve Agent logging by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/136
* Add nicer error message for agents without tools by @jkornblum-nv in https://github.com/NVIDIA/AIQToolkit/pull/146
* Add `colorama` to core dependency by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/149
* Rename packages  agentiq -> aiqtoolkit by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/152
* Rename AIQ_COMPONENT_NAME, remove unused COMPONENT_NAME by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/153
* Group wheels under a common `aiqtoolkit` directory by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/154
* Fix wheel upload wildcards by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/155
* Support Python `3.11` for AgentIQ by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/148
* fix pydantic version incompatibility, closes #74 by @zac-wang-nv in https://github.com/NVIDIA/AIQToolkit/pull/159
* Rename AgentIQ to Agent Intelligence Toolkit by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/160
* Create config file symlink with `aiq workflow create` command by @mpenn in https://github.com/NVIDIA/AIQToolkit/pull/166
* Rename generate/stream/full to generate/full and add filter_steps parameter by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/164
* Add support for environment variable interpolation in config files by @mpenn in https://github.com/NVIDIA/AIQToolkit/pull/157
* UI submodule rename by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/168
* Consistent Trace Nesting in Parallel Function Calling  by @dnandakumar-nv in https://github.com/NVIDIA/AIQToolkit/pull/162
* Fix broken links in examples documentation by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/177
* Remove support for Python `3.13` by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/178
* Add transitional packages by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/181
* Add a tunable RAG evaluator by @liamy-nv in https://github.com/NVIDIA/AIQToolkit/pull/110
* CLI Documentation fixes in remote registry configuration section by @mpenn in https://github.com/NVIDIA/AIQToolkit/pull/184
* Fix uploading of transitional packages by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/187
* Update `AIQChatRequest` to support image and audio input by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/182
* Fix hyperlink ins the simple_calculator README by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/188
* Add support for fine-grained tracing using W&B Weave by @ayulockin in https://github.com/NVIDIA/AIQToolkit/pull/170
* Fix typo in CPR detected by co-pilot by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/190
* Note the name change in the top-level documentation and README.md by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/163
* fix typo in evaluate documentation for max_concurrency by @soumilinandi in https://github.com/NVIDIA/AIQToolkit/pull/191
* Fix a typo in the weave README by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/195
* Update simple example `eval` dataset by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/200
* Config option to specify the intermediate step types in workflow_output.json by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/198
* Update the Judge LLM settings in the examples to avoid retries by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/204
* Make `opentelemetry` and `phoenix` as optional dependencies by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/167
* Support user-defined HTTP request metadata in workflow tools. by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/130
* Check if request is present before setting attributes by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/209
* Add the alert triage agent example by @hsin-c in https://github.com/NVIDIA/AIQToolkit/pull/193
* Updating ui submodule by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/211
* Fix plugin dependencies by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/208
* [FEA]add profiler agent to the examples folder by @zac-wang-nv in https://github.com/NVIDIA/AIQToolkit/pull/120
* Regenerate `uv.lock`, cleaned up `pyproject.toml` for profiler agent example and fixed broken link in `README` by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/210
* Removed `disable=unused-argument` from pylint checks by @Hritik003 in https://github.com/NVIDIA/AIQToolkit/pull/186
* Exception handling for discovery_metadata.py by @VictorYudin in https://github.com/NVIDIA/AIQToolkit/pull/215
* Fix incorrect eval output config access by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/219
* Treat a tagged commit the same as a nightly build by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/217
* Feature/add aiqtoolkit UI submodule by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/214
* Add a CLI command to list all tools available via the MCP server by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/221
* For remote evaluation, workflow config is not needed by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/225
* Move configurable parameters from env vars to config file by @hsin-c in https://github.com/NVIDIA/AIQToolkit/pull/222
* Fix vulnerabilities in the alert triage agent example by @hsin-c in https://github.com/NVIDIA/AIQToolkit/pull/227
* Add e2e test for the alert triage agent by @hsin-c in https://github.com/NVIDIA/AIQToolkit/pull/226
* Fix remaining nSpect vulnerabilities for `1.1.0` by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/229
* Remove redundant span stack handling and error logging by @dnandakumar-nv in https://github.com/NVIDIA/AIQToolkit/pull/231
* Feature/add aiqtoolkit UI submodule by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/234
* Fix `Dockerfile` build failure for `v1.1.0-rc3` by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/240
* Bugfix for alert triage agent to run in python 3.11 by @hsin-c in https://github.com/NVIDIA/AIQToolkit/pull/244
* Misc example readme fixes by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/242
* Fix multiple documentation and logging bugs for `v1.1.0-rc3` by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/245
* Consolidate MCP client and server docs, examples by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/246
* Update version of llama-index to 0.12.21 by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/257
* Fix environment variable interpolation with console frontend by @mpenn in https://github.com/NVIDIA/AIQToolkit/pull/255
* [AIQ][25.05][RC3] Example to showcase Metadata support by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/256
* mem: If conversation is not provided build it from memory by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/253
* Documentation restructure by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/189
* Prompt engineering to force `ReAct` agent to use memory for `simple_rag` example by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/260
* simple-calculator: Additional input validation by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/259
* Removed simple_mcp example by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/266
* Adding reference links to examples in README. by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/265
* mcp-client.md: Add a note to check that the MCP time-service is running by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/267
* Remove username from `README` log by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/271
* Enhance error handling in MCP tool invocation by @mpenn in https://github.com/NVIDIA/AIQToolkit/pull/263
* Resolves a linting error in MCP tool by @mpenn in https://github.com/NVIDIA/AIQToolkit/pull/274
* Fix long-term memory issues of `semantic_kernel` example by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/270
* Update to reflect new naming guidelines by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/258
* Updating submodule that fixes UI broken links by @ericevans-nv in https://github.com/NVIDIA/AIQToolkit/pull/273
* Change the example input for `Multi Frameworks` example by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/277
* Fix intermediate steps parents when the parent is a Tool by @mdemoret-nv in https://github.com/NVIDIA/AIQToolkit/pull/269
* Set mcp-proxy version in the sample Dockerfile to 0.5 by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/278
* Add an FAQ document by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/275
* Fix missing tool issue with `profiler_agent` example by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/279
* Add missing `telemetry` dependency to `profiler_agent` example by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/281
* eval-readme: Add instruction to copy the workflow output before re-runs by @AnuradhaKaruppiah in https://github.com/NVIDIA/AIQToolkit/pull/280
* Add additional notes for intermittent long-term memory issues in examples by @yczhang-nv in https://github.com/NVIDIA/AIQToolkit/pull/282
* Run tests on all supported versions of Python by @dagardner-nv in https://github.com/NVIDIA/AIQToolkit/pull/283
* Fix the intermediate steps span logic to work better with nested coroutines and tasks by @mdemoret-nv in https://github.com/NVIDIA/AIQToolkit/pull/285

### New Contributors
* @dagardner-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/7
* @yczhang-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/9
* @gzitzlsb-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/25
* @atalhens made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/28
* @avoroshilov made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/42
* @balvisio made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/22
* @ericevans-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/64
* @dnandakumar-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/70
* @wenqiglantz made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/36
* @VictorYudin made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/133
* @Hritik003 made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/129
* @jkornblum-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/146
* @zac-wang-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/159
* @mpenn made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/166
* @liamy-nv made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/110
* @ayulockin made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/170
* @soumilinandi made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/191
* @hsin-c made their first contribution in https://github.com/NVIDIA/AIQToolkit/pull/193

## [1.1.0a1] - 2025-04-05

### Added
- Added CORS configuration for the FastAPI server
- Added support for customizing evaluation outputs and uploading results to remote storage

### Fixed
- Fixed `aiq serve` when running the `simple_rag` workflow example
- Added missing `platformdirs` dependency to `aiqtoolkit` package

## [1.0.0] - 2024-12-04

### Added

- First release.
