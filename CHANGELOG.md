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

## [1.2.1] - 2025-08-20
### 📦 Overview
This is a documentation only release, there are no code changes in this release.

### 📜 Full Change Log
* Add a version switcher to the documentation builds https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/681

## [1.2.0] - 2025-08-20
### 📦 Overview
The NeMo Agent toolkit, formerly known as Agent Intelligence (AIQ) toolkit, has been renamed to align with the NVIDIA NeMo family of products. This release brings significant new capabilities and improvements across authentication, resource management, observability, and developer experience. The toolkit continues to offer backwards compatibility, making the transition seamless for existing users.

While NeMo Agent Toolkit is designed to be compatible with the previous version, users are encouraged to update their code to follow the latest conventions and best practices. Migration instructions are provided in the [migration guide](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/resources/migration-guide.md).

### 🚨 Breaking Changes
* Remove outdated/unsupported devcontainer by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/626
* Rename `aiq` namespace to `nat` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/618
* Update `AIQ` to `NAT` in documentation and comments by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/614
* Remove `AIQ` prefix from class and function names by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/606
* Rename aiqtoolkit packages to nvidia-nat by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/598
* Observability redesign to reduce dependencies and improve flexibility by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/379

### 🚀 Notable Features and Improvements
* [Authentication for Tool Calling](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/reference/api-authentication.md): Implement robust authentication mechanisms that enable secure and configurable access management for tool invocation within agent workflows.
* [Test Time Compute](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/reference/test-time-compute.md): Dynamically reallocate compute resources after model training, allowing agents to optimize reasoning, factual accuracy, and system robustness without retraining the base model.
* [Sizing Calculator](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/workflows/sizing-calc.md): Estimate GPU cluster requirements to support your target number of users and desired response times, simplifying deployment planning and scaling.
* [Object Store Integration](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/extend/object-store.md): Connect and manage data through supported object stores, improving agent extensibility and enabling advanced data workflows.
* [Enhanced Cursor Rules](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/tutorials/build-a-demo-agent-workflow-using-cursor-rules.md): Build new workflows or extend existing ones by leveraging cursor rules, making agent development faster and more flexible.
* [Interactive Notebooks](https://github.com/NVIDIA/NeMo-Agent-Toolkit/tree/release/1.2/examples/notebooks): Access a suite of onboarding and example notebooks to accelerate agent workflow development, testing, and experimentation.
* [Observability Refactor](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/docs/source/workflows/observe/index.md): Onboard new observability and monitoring platforms more easily, and take advantage of improved plug-in architecture for workflow inspection and analysis.
* [Examples Reorganization](https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.2/examples/README.md): Organize examples by functionality, making it easier to find and use the examples.

### 📜 Full Change Log
* Use consistent casing for ReAct agent by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/293
* Update alert triage agent's prompt by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/297
* Move Wikipedia search to separate file by @jkornblum-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/237
* Release documentation fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/300
* Add a `pyproject.toml` to `simple_rag` example allowing for declared dependencies by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/284
* Update version in develop, in prep for the next release by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/294
* Add field validation for the evaluate API by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/311
* Intermediate steps: evaluation fix by @titericz in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/312
* Fix or silence warnings emitted by tests by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/305
* Add documentation for `load_workflow()` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/306
* Adding pytest-pretty for nice test outputs by @benomahony in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/194
* feat(telemetry): add langfuse and langsmith telemetry exporters #233 by @briancaffey in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/235
* Check links in markdown files by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/323
* Eval doc updates by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/322
* Add unit tests for the alert triage agent example by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/252
* Add support for AWS Bedrock LLM Provider by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/238
* Add missing import in `load_workflow` documentation by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/329
* propose another solution to problem[copy] by @LunaticMaestro in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/298
* Support additional_instructions by @gfreeman-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/302
* Update installing.md by @manny-pi in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/316
* Add an async version of the /generate endpoint by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/315
* Update trajectory eval documentation by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/338
* Rename test mode to offline mode by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/343
* Simplify offline mode with `aiq eval` by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/344
* fix mcp client schema creation in flat lists by @slopp in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/346
* Refactor for better prompt and tool description organization by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/350
* Extend `IntermediateStep` to support tool schemas in tool calling LLM requests by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/357
* Fix AttributeError bug for otel_telemetry_exporter by @ZhongxuanWang in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/335
* Update `OpenAIModelConfig` to support `stream_usage` option by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/328
* Rename to NeMo Agent toolkit by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/359
* fix(phoenix): set project name when using phoenix telemetry exporter (#337) by @briancaffey in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/347
* Account for the "required fields" list in the mcp_input_schema by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/360
* Provide a config to pass the complete dataset entry as an EvalInputItem field to evaluators by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/355
* Simplify custom evaluator definition by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/358
* Add Patronus OTEL Exporter by @hersheybar in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/341
* Expand Alert Triage Agent Offline Dataset by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/369
* Add Custom Classification Accuracy Evaluator for the Alert Triage Agent by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/373
* Add `Cursor rules` to improve Cursor support for development by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/319
* Added LLM retry logic to handle rate limiting LLM without frequent Exception by @liamy-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/368
* Fixes Function and LambdaFunction classes to push active function instance names by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/374
* TunableRagEvaluator: Re-enable inheriting from the base abc by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/375
* Add Job ID Appending to Output Directories and Maximum Folders Threshold by @ZhongxuanWang in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/331
* Add support for custom functions in bottleneck analysis by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/380
* Persist chat conversation ID for workflow tool usage by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/326
* Add support for Weave evaluation by @ayulockin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/264
* Update the information displayed in the Weave Eval dashboard by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/390
* Allow non-json string outputs for workflows that use unstructured datasets by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/396
* Add aws region config for s3 eval uploads by @munjalp6 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/397
* add support for union types in mcp client by @cheese-head in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/372
* Add percentile computation (p90, p95, p99) to profiling by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/398
* Ragas custom evaluation field in evaluator by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/400
* Reorganize the examples into categories and improve re-use of example components by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/411
* Improve descriptions in top level examples README.md by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/415
* Add ragaai catalyst exporters by @vishalk-06 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/395
* Update MCP version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/417
* feature request: Add galileo tracing workflow by @franz101 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/404
* Update index.md by @sugsharma in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/420
* Windows compatibility for temp file handling by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/423
* Sizing calculator to estimate the number of GPU for a target number of users by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/399
* Update and move W&B Weave Redact PII example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/424
* Refactor IntermediateStep `parent_id` for clarification by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/330
* Add Cursor rules for latinisms by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/426
* Resolve examples organization drift by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/429
* NeMo Agent rename by @lvojtku in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/422
* Removes redundant config variable from Retry Agent Function. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/447
* Add otelcollector doc and example by @slopp in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/451
* Improve error logging during workflow initialization failure by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/464
* Added an AIQToolkit function that can be invoked to perform a simple completions task, given a natural language prompt. by @sayalinvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/460
* Improve MCP error logging with connection failures by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/470
* Enable testing tools in isolation by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/391
* Refactor examples to improve discoverability and improve uniformity by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/476
* Add Inference Time Scaling Module by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/381
* Refactor Agno Personal Finance Function and Update Configuration by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/477
* Add object store by @balvisio in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/299
* Observability redesign to reduce dependencies and improve flexibility by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/379
* Adding OpenAI Chat Completions API compatibility by @dfagnou in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/421
* Enhance code execution sandbox with improved error handling and debugging by @vikalluru in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/409
* Fixing inheritance on the OTel collector exporter and adding project name by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/481
* Refactor retry mechanism and update retry mixin field config by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/480
* Integrate latest `RetryMixin` fixes with `aiqtoolkit_agno` subpackage by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/483
* Fix incorrect file paths in simple calculator example by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/482
* Documentation edits for sizing calculator by @lvojtku in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/436
* feat(redis): add redis memory backend and redis memory example #376 by @briancaffey in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/377
* Improve error handling and recovery mechanisms in agents by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/418
* Update `git clone` under `/doc` folder to point to `main` branch by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/484
* Pin `datasets` version in toplevel `pyproject.toml` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/487
* Fix `otelcollector` to ensure project name is added to `OtelSpan` resource + added weave cleanup logic by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/488
* Fix shared field reference bug in `TypedBaseModel` inheritance by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/489
* Streamlining API Authentication by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/251
* Add experimental decorator to auth and ITS strategy methods by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/493
* Unify examples README structure by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/485
* Cleanup authorization settings to remove unnecessary options by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/495
* Move `WeaveMixin._weave_calls` to `IsolatedAttribute` to avoid cleanup race conditions by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/499
* Set SCM versioning for `text_file_ingest` allowing it to be built in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/501
* Update PII example to improve user experience and `WeaveExporter` robustness by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/502
* Fix `pyproject.toml` for `text_file_ingest` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/505
* Update `ci/checks.sh` to run all of the same checks performed by CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/506
* Suppress stack trace in error message in `ReActOutputParserException` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/507
* Clarify intermediate output formatting in agent tool_calling example by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/504
* Fix fastapi endpoint for plot_charts by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/508
* Fix UI docs to launch the simple calculator by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/511
* Update prerequisite and system prompt of `redis` memory example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/510
* Fix HITL `por_to_jiratickets` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/515
* Relax overly restrictive constraints on AIQChatRequest model by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/512
* Fix file paths for simple_calculator_eval by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/517
* Fix: getting_started docker containers build with added compiler dependency by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/518
* Documentation: Specify minimum uv version by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/520
* Fix Simple Calculator HITL Example. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/519
* Fix outdated path in `pyproject.toml` in `text_file_ingest` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/524
* Fix issue where aiq fails for certain log levels by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/523
* Update catalyst readme document by @vishalk-06 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/492
* Fix outdated file references under `/examples` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/526
* Misc Documentation cleanups by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/527
* Misc cleanups/fixes for `installing.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/528
* Fix: file path references in examples and docs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/536
* Resolve batch flushing failure during `SpanExporter` cleanup by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/532
* Fix typos in the observability info commands by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/529
* Publish the linear fit data in the CalcRunner Output by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/498
* Restructure example README to fully align with reorganization by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/539
* Add example for Vulnerability Analysis for Container Security Blueprint by @ashsong-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/530
* Misc cleanups for `docs/source/quick-start/launching-ui.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/537
* Fix grammar error in uninstall help string by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/540
* Fix: custom routing example typos and output clarification by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/541
* Update the UI submodule to adopt fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/543
* Fix: Examples README output clarifications; installation command by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/542
* Ensure type system and functional behavior are consistent for `to_type` specifications by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/513
* Documentation: update memory section to include redis; fix code references by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/544
* Update the dataset in the swe-bench README by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/546
* Fix alert triage agent documentation on system output by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/545
* Fix several dependency and documentation issues under `/examples` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/547
* Fixes to the `add-tools-to-a-workflow.md` tutorial by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/548
* Example: update `swe_bench` README to reflect output changes by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/550
* Update Cursor rules and documentations to remove unnecessary installation checks by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/549
* Update `object_store` example to use NVIDIA key instead of missing OPENAI key by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/552
* Remove deprecated code usage in the `por_to_jiratickets` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/557
* Fix `simple_auth` link to UI repository by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/553
* Update the LangSmith environment variable names in `simple_calculator_observability` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/558
* Improvements to extending telemetry exporters docs by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/554
* Misc cleanups for `create-a-new-workflow.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/556
* Reduce the number of warnings logged while running the `getting_started` examples by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/563
* Update observability system documentation to reflect modern architecture and remove snippets by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/562
* Add docker container for oauth server to fix configuration issues by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/561
* Ensure imports are lazily loaded in plugins improving startup time by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/564
* General improvements to `observe-workflow-with-catalyst.md` to improve experience by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/555
* Update Simple Auth Example Config File Path by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/566
* Convert `cursor_rules_demo` GIF files to Git LFS by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/567
* UI Submodule Update by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/568
* Restructure agents documentation by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/569
* Increase package/distro resolution in `DiscoveryMetadata` to improve utility of `aiq info components` by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/574
* Minor cleanups for the `run-workflows.md` document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/572
* Add CI check for path validation within repository by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/573
* Improvements to observability plugin documentation by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/578
* Fixes for `adding-an-authentication-provider.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/579
* Minor cleanups for `sizing-calc.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/577
* minor doc update to pass lint by @gfreeman-nvidia in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/582
* Fixing missing space preventing proper render of snippet in markdown by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/581
* Fixing wrong override usage to make it compatible with py 3.11 by @vikalluru in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/585
* Updating UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/588
* Fixes for `api-authentication.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/583
* Object Store code, documentation, and example improvements by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/587
* Fix module discovery errors when publishing with registry handlers by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/592
* Update logging levels in `ProcessingExporter`and `BatchingProcessor` to reduce shutdown noise by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/589
* Update template to use `logger` instead of `print` by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/590
* Fix fence indenting, remove ignore pattern by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/594
* Remove Unused Authentication Components from Refactor by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/596
* Minor cleanup to `using-local-llms.md` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/595
* Merge Post VDR changes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/597
* Rename aiqtoolkit packages to nvidia-nat by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/598
* Rename Inference Time Scaling to Test Time Compute by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/600
* CI: upload script updated to set the artifactory path's top level dir by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/602
* Rename ITS tool functions to TTC tool functions by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/605
* Fix the artifactory component name to aiqtoolkit for all packages by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/603
* Fix Pylint in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/609
* Remove `AIQ` prefix from class and function names by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/606
* Add support for synchronous LangChain tool calling by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/612
* Send Conversation ID with WebSocket Messages by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/613
* Adds support for MCP server /health endpoint, custom routes and a client `mcp ping` command  by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/576
* Updating UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/622
* Rename `aiq` namespace to `nat` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/618
* Remove outdated/unsupported devcontainer by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/626
* Use issue types instead of title prefixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/619
* Fixes to `weave` telemetry exporter to ensure traces are properly sent to Weave by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/627
* Apply work-around for #621 to the gitlab ci scripts by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/630
* Revert unintended change to `artifactory_upload.sh` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/631
* Bugfix (Object Store): remove unnecessary S3 refs in config; fix mysql upload script by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/628
* Refactor embedder client structure for LangChain and Llama Index. by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/634
* Documentation: Update Using Local LLMs by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/623
* Align WebSocket Workflow Output with HTTP Output by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/635
* Update `AIQ` to `NAT` in documentation and comments by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/614
* Documentation(Providers): Surface LLM; add Embedders and Retrievers by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/625
* CI: improve path-check utility; fix broken links; add more path check rules by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/601
* Fix broken `additional_instructions` options for `ReWOO` agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/640
* Updating ui submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/641
* Fix symlink structure to be consistent across all examples by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/642
* Fix: add missing uv.source for `simple_calculator_hitl` by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/638
* Enable datasets with custom formats by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/615
* Update uv.lock by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/644
* Run CI for commits to the release branch by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/645
* Add a note that the dataset needs to be uploaded to the S3 bucket by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/646
* Add UI documentation links and installation instructions by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/647
* Consolidate CI pipelines by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/632
* Install git-lfs in docs CI stage by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/648
* Fix `aiq` compatibility by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/651
* Bugfix: Align Python Version Ranges by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/655
* Docs: Add Migration Guide by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/653
* Update third-party-license files for v1.2 by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/657
* Add notebooks to show users how to get started with the toolkit and build agents by @cdgamarose-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/656
* Remove redundant prefix from directory names by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/665
* Docs: Add Upgrade Fix to Troubleshooting by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/659
* Enhance README with badges, installation, instructions, and a roadmap by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/654
* Adding `.nspect-allowlist.toml` to remediate false positives found by scanner by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/668
* Fix: Remove `pickle` from MySQL-based Object Store by @willkill07 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/669
* Enable `BUILD_NAT_COMPAT` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/670
* Fix paths for compatibility packages by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/672
* Add missing compatibility package for `aiqtoolkit-weave` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/674
* Add chat_history to the context of ReAct and ReWOO agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/673

### 🙌 New Contributors
* @titericz made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/312
* @benomahony made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/194
* @briancaffey made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/235
* @LunaticMaestro made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/298
* @gfreeman-nvidia made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/302
* @manny-pi made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/316
* @slopp made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/346
* @ZhongxuanWang made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/335
* @hersheybar made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/341
* @munjalp6 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/397
* @cheese-head made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/372
* @vishalk-06 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/395
* @franz101 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/404
* @sugsharma made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/420
* @lvojtku made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/422
* @sayalinvidia made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/460
* @dfagnou made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/421
* @vikalluru made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/409
* @ashsong-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/530

## [1.1.0] - 2025-05-16
### Key Features
- Full MCP (Model Context Protocol) support
- Weave tracing
- Agno integration
- ReWOO Agent
- Alert Triage Agent Example

### What's Changed
* Have the examples README point to the absolute path by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/4
* Set initial version will be 1.0.0 by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/7
* Update `examples/simple_rag/README.md` to verify the installation of `lxml` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/9
* Use a separate README for pypi by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/10
* Document the need to install from source to run examples by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/8
* Fixing broken links by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/14
* Cleanup readmes by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/15
* Pypi readme updates by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/16
* Final 1.0.0 cleanup by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/18
* Add subpackage readmes redirecting to the main package by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/20
* Update README.md by @gzitzlsb-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/25
* Fix #27 Documentation fix by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/28
* Fix #29 - Simple_calculator example throws error - list index out of range when given subtraction by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/31
* Fix: #32 Recursion Issue by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/33
* "Sharing NVIDIA AgentIQ Components" docs typo fix by @avoroshilov in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/42
* First pass at setting up issue templates by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/6
* Provide a cleaner progress bar when running evaluators in parallel by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/38
* Setup GHA CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/46
* Switch UI submodule to https by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/53
* gitlab ci pipeline cleanup by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/54
* Allow str or None for retriever description by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/55
* Fix case where res['categories'] = None by @balvisio in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/22
* Misc CI improvements by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/56
* CI Documentation improvements by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/24
* Add missing `platformdirs` dependency by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/62
* Fix `aiq` command error when the parent directory of `AIQ_CONFIG_DIR` does not exist by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/63
* Fix broken image link in multi_frameworks documentation by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/61
* Updating doc string for AIQSessionManager class. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/64
* Fix ragas evaluate unit tests by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/68
* Normalize Gannt Chart Timestamps in Profiler Nested Stack Analysis by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/70
* Scripts for running CI locally by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/59
* Update types for `topic` and `description` attributes in  `AIQRetrieverConfig` to allow `None` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/76
* Add support for customizing output and uploading it to remote storage (S3 bucket) by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/71
* Support ARM in CI by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/73
* Allow overriding configuration values not set in the YAML by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/85
* Fix bug where `--workers` flag was being ignored by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/88
* Adding Cors config for api server by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/89
* Update changelog for 1.1.0a1 alpha release by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/90
* Updated changelog with another bug fix by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/93
* Adjust how the base_sha is passed into the workflow by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/81
* Changes for evaluating remote workflows by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/57
* Fix a bug in our pytest plugin causing test coverage to be under-reported by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/105
* Docker container for AgentIQ by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/87
* Modify JSON serialization to handle non-serializable objects by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/106
* Upload nightly builds and release builds to pypi by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/112
* Ensure the nightly builds have a unique alpha version number by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/115
* Ensure tags are fetched prior to determining the version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/116
* Fix CI variable value by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/117
* Use setuptools_scm environment variables to set the version by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/118
* Only set the setuptools_scm variable when performing a nightly build by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/119
* Add a release PR template by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/123
* Add an async /evaluate endpoint to trigger evaluation jobs on a remote cluster by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/109
* Update /evaluate endpoint doc by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/126
* Add function tracking decorator and update IntermediateStep by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/98
* Fix typo in aiq.profiler.decorators by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/132
* Update the start command to use `validate_schema` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/82
* Document using local/self-hosted models by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/101
* added Agno integration by @wenqiglantz in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/36
* MCP Front-End Implementation by @VictorYudin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/133
* Make kwargs optional to the eval output customizer scripts by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/139
* Add an example that shows simple_calculator running with a MCP service. by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/131
* add `gitdiagram` to README by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/141
* Updating HITL reference guide to instruct users to toggle ws mode and… by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/142
* Add override option to the eval CLI command by @Hritik003 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/129
* Implement ReWOO Agent by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/75
* Fix type hints and docstrings for `ModelTrainer` by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/107
* Delete workflow confirmation check in CLI - #114 by @atalhens in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/137
* Improve Agent logging by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/136
* Add nicer error message for agents without tools by @jkornblum-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/146
* Add `colorama` to core dependency by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/149
* Rename packages  agentiq -> aiqtoolkit by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/152
* Rename AIQ_COMPONENT_NAME, remove unused COMPONENT_NAME by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/153
* Group wheels under a common `aiqtoolkit` directory by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/154
* Fix wheel upload wildcards by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/155
* Support Python `3.11` for AgentIQ by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/148
* fix pydantic version incompatibility, closes #74 by @zac-wang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/159
* Rename AgentIQ to Agent Intelligence Toolkit by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/160
* Create config file symlink with `aiq workflow create` command by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/166
* Rename generate/stream/full to generate/full and add filter_steps parameter by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/164
* Add support for environment variable interpolation in config files by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/157
* UI submodule rename by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/168
* Consistent Trace Nesting in Parallel Function Calling  by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/162
* Fix broken links in examples documentation by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/177
* Remove support for Python `3.13` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/178
* Add transitional packages by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/181
* Add a tunable RAG evaluator by @liamy-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/110
* CLI Documentation fixes in remote registry configuration section by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/184
* Fix uploading of transitional packages by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/187
* Update `AIQChatRequest` to support image and audio input by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/182
* Fix hyperlink ins the simple_calculator README by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/188
* Add support for fine-grained tracing using W&B Weave by @ayulockin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/170
* Fix typo in CPR detected by co-pilot by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/190
* Note the name change in the top-level documentation and README.md by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/163
* fix typo in evaluate documentation for max_concurrency by @soumilinandi in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/191
* Fix a typo in the weave README by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/195
* Update simple example `eval` dataset by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/200
* Config option to specify the intermediate step types in workflow_output.json by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/198
* Update the Judge LLM settings in the examples to avoid retries by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/204
* Make `opentelemetry` and `phoenix` as optional dependencies by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/167
* Support user-defined HTTP request metadata in workflow tools. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/130
* Check if request is present before setting attributes by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/209
* Add the alert triage agent example by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/193
* Updating ui submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/211
* Fix plugin dependencies by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/208
* [FEA]add profiler agent to the examples folder by @zac-wang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/120
* Regenerate `uv.lock`, cleaned up `pyproject.toml` for profiler agent example and fixed broken link in `README` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/210
* Removed `disable=unused-argument` from pylint checks by @Hritik003 in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/186
* Exception handling for discovery_metadata.py by @VictorYudin in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/215
* Fix incorrect eval output config access by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/219
* Treat a tagged commit the same as a nightly build by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/217
* Feature/add aiqtoolkit UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/214
* Add a CLI command to list all tools available via the MCP server by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/221
* For remote evaluation, workflow config is not needed by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/225
* Move configurable parameters from env vars to config file by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/222
* Fix vulnerabilities in the alert triage agent example by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/227
* Add e2e test for the alert triage agent by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/226
* Fix remaining nSpect vulnerabilities for `1.1.0` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/229
* Remove redundant span stack handling and error logging by @dnandakumar-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/231
* Feature/add aiqtoolkit UI submodule by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/234
* Fix `Dockerfile` build failure for `v1.1.0-rc3` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/240
* Bugfix for alert triage agent to run in python 3.11 by @hsin-c in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/244
* Misc example readme fixes by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/242
* Fix multiple documentation and logging bugs for `v1.1.0-rc3` by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/245
* Consolidate MCP client and server docs, examples by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/246
* Update version of llama-index to 0.12.21 by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/257
* Fix environment variable interpolation with console frontend by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/255
* [AIQ][25.05][RC3] Example to showcase Metadata support by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/256
* mem: If conversation is not provided build it from memory by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/253
* Documentation restructure by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/189
* Prompt engineering to force `ReAct` agent to use memory for `simple_rag` example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/260
* simple-calculator: Additional input validation by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/259
* Removed simple_mcp example by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/266
* Adding reference links to examples in README. by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/265
* mcp-client.md: Add a note to check that the MCP time-service is running by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/267
* Remove username from `README` log by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/271
* Enhance error handling in MCP tool invocation by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/263
* Resolves a linting error in MCP tool by @mpenn in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/274
* Fix long-term memory issues of `semantic_kernel` example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/270
* Update to reflect new naming guidelines by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/258
* Updating submodule that fixes UI broken links by @ericevans-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/273
* Change the example input for `Multi Frameworks` example by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/277
* Fix intermediate steps parents when the parent is a Tool by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/269
* Set mcp-proxy version in the sample Dockerfile to 0.5 by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/278
* Add an FAQ document by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/275
* Fix missing tool issue with `profiler_agent` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/279
* Add missing `telemetry` dependency to `profiler_agent` example by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/281
* eval-readme: Add instruction to copy the workflow output before re-runs by @AnuradhaKaruppiah in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/280
* Add additional notes for intermittent long-term memory issues in examples by @yczhang-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/282
* Run tests on all supported versions of Python by @dagardner-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/283
* Fix the intermediate steps span logic to work better with nested coroutines and tasks by @mdemoret-nv in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/285

### New Contributors
* @dagardner-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/7
* @yczhang-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/9
* @gzitzlsb-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/25
* @atalhens made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/28
* @avoroshilov made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/42
* @balvisio made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/22
* @ericevans-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/64
* @dnandakumar-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/70
* @wenqiglantz made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/36
* @VictorYudin made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/133
* @Hritik003 made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/129
* @jkornblum-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/146
* @zac-wang-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/159
* @mpenn made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/166
* @liamy-nv made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/110
* @ayulockin made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/170
* @soumilinandi made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/191
* @hsin-c made their first contribution in https://github.com/NVIDIA/NeMo-Agent-Toolkit/pull/193

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
