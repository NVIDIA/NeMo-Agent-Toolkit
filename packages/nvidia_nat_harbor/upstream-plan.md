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

# Harbor Upstream Plan for `nvidia-nat-harbor`

This plan separates generic Harbor framework improvements from NAT-specific
runtime and evaluator integration. The goal is to avoid long-lived Harbor fork
drift while keeping NAT behavior inside `nvidia-nat-harbor` unless Harbor
already owns the abstraction.

## Principles

- Upstream behavior that is useful for any Harbor agent, verifier, or benchmark.
- Keep NAT runtime contracts, evaluator dispatch, and sample coercion in
  `nvidia-nat-harbor`.
- Make local execution easy to use, but describe it as developer convenience,
  not benchmark isolation.
- Preserve shell-mode Harbor workflows while adding inline/library-mode paths.

## Terms

| Term | Meaning |
|---|---|
| Local environment | Host-local Harbor environment backend selected by `--env local` once upstreamed. |
| Shell compatibility mode | Existing Harbor `nemo-agent` subprocess/wrapper path plus task `tests/test.sh`. |
| Library mode | NAT workflow execution in-process through the active Harbor Python. |
| Inline verifier | Harbor verifier object that scores in-process through a verifier import hook. |
| Script bridge | Compatibility path where task `tests/test.sh` invokes `nat_harbor.verifier.bridge_runner`. |

## Upstream Priority Table

| Priority | Upstream item | Why it matters | NAT impact while missing |
|---|---|---|---|
| P0 | Generic verifier extension hook (`VerifierFactory`, `VerifierConfig.import_path`, `VerifierConfig.kwargs`, `--verifier-import-path`, `--verifier-kwarg`) | Allows NAT to provide `ATIFInlineVerifier` without modifying Harbor verifier internals. | We must carry a Harbor side branch or patch Harbor installs to run inline verifier examples. |
| P1 | First-class `--env local` and generic local environment behavior | Removes the temporary `--env docker` enum workaround and makes host-local Harbor execution explicit. | README commands need `--env docker` plus `--environment-import-path ...LocalEnvironment`, which is confusing and easy to misread as Docker isolation. |
| P1 | Local install policy primitive for installed agents | Gives local runs a safe default that skips host mutation unless explicitly allowed. | NAT has to carry local install policy handling in its Harbor agent wrapper. |
| P2 | Generic agent run-mode extension point for installed agents | Lets Harbor support library/in-process execution without baking NAT evaluator details into core. | NAT keeps overriding `nemo-agent` behavior to select shell mode vs library mode. |
| P2 | Harbor-owned `nemo-agent` library mode, if maintainers accept NAT runtime ownership | Could make library mode a first-class Harbor `nemo-agent` feature instead of a package-local extension. | `DefaultNemoInlineRunner` remains in `nvidia-nat-harbor`, and Harbor `nemo-agent` stays shell-first. |
| P3 | Shared docs/examples for local env and imported verifiers | Helps non-NAT integrations use the same extension points. | NAT docs have to explain generic Harbor concepts alongside NAT-specific setup. |

## Upstream Area 1: Local Environment

### Upstream To Harbor

- Add first-class `--env local` support in Harbor CLI/type validation.
- Provide a generic local environment backend aligned with Harbor environment
  abstractions:
  - path mapping for `/app`, `/workspace`, `/logs`, and `/tests`
  - lifecycle setup/cleanup
  - upload/download behavior
  - preflight checks
  - clear non-sandbox semantics
- Add a generic local install policy primitive usable by installed agents:
  - `skip`: assume dependencies are already available on the host
  - `allow`: permit local installation during agent setup
- Optionally standardize policy metadata emission so integrations can record
  whether setup was skipped or executed.

### Keep In `nvidia-nat-harbor`

- NAT-specific local run examples and README commands.
- Any temporary path-translation workaround needed before Harbor owns local env.

### Acceptance Criteria

- `harbor run --env local` works without enum workarounds.
- Local env artifacts land under the trial directory with the same logical
  layout expected by existing agents and verifiers.
- Local env docs state that host processes are executed directly and are not
  sandboxed.
- Existing shell-mode Harbor runs still work.

## Upstream Area 2: Verifier Extension Hook

### Upstream To Harbor

- Keep verifier creation behind a generic factory.
- Expose verifier extension wiring in config and CLI:
  - `VerifierConfig.import_path`
  - `VerifierConfig.kwargs`
  - `--verifier-import-path`
  - `--verifier-kwarg`
- Preserve the Harbor verifier contract for all verifier implementations:
  - verifier output directory is provided consistently
  - `reward.txt` is consumed consistently
  - structured reward/details files are allowed but optional
  - errors propagate as trial verifier failures
  - single-step and multi-step verification use the same factory path

### Landed Minimal Harbor Hook

- `VerifierFactory` in Harbor verifier runtime.
- `VerifierConfig.import_path` and `VerifierConfig.kwargs`.
- CLI flags:
  - `--verifier-import-path`
  - `--verifier-kwarg`
- Trial verification constructs verifiers through `VerifierFactory` in both
  single-step and multi-step paths.

### Keep In `nvidia-nat-harbor`

- `ATIFInlineVerifier`.
- NAT evaluator dispatch:
  - `evaluate_atif_fn` resolution
  - NAT config loading
  - evaluator-name selection
  - custom callable evaluator lane
  - ATIF sample coercion conventions

### Acceptance Criteria

- A non-Harbor verifier class can be loaded by import path and receive kwargs.
- Imported verifiers can write the same reward artifacts as script verifiers.
- Harbor result aggregation is identical for built-in, script, and imported
  verifiers.
- NAT inline verifier examples do not require patching Harbor inside `.venv` or
  `site-packages`.

## Upstream Area 3: Agent Run Modes

The concrete NAT inline runner is the boundary that needs the most care. Harbor
already owns `nemo-agent`, but in-process NAT workflow execution depends on NAT
runtime contracts. There are two acceptable upstream shapes:

1. Harbor owns a generic run-mode extension point, while `nvidia-nat-harbor`
   provides the concrete NAT library-mode runner.
2. Harbor's `nemo-agent` owns library mode directly, if Harbor maintainers accept
   the NAT runtime dependency and maintenance burden.

### Candidate Harbor Upstreams

- Generic agent run-mode selection and kwargs plumbing.
- Local-env defaults for installed agents.
- Shell-mode fallback behavior for parity debugging.
- Quality-of-life improvements in Harbor `nemo-agent` that are not NAT
  evaluator-specific:
  - local install policy wiring
  - clearer setup metadata
  - wrapper/runtime error reporting

### Keep In `nvidia-nat-harbor`

- `DefaultNemoInlineRunner` unless Harbor accepts direct ownership.
- NAT workflow loading and invocation details.
- ATIF trajectory construction from NAT intermediate steps.
- Inline environment overlay serialization for process-global `os.environ`.
- NAT-specific integration tests and examples.

### Acceptance Criteria

- Shell mode remains available and backward compatible.
- Library mode works for local Harbor runs without invoking the wrapper
  subprocess.
- Concurrent inline trials cannot read another trial's temporary environment
  overlay.
- Missing trajectory/intermediate-step output still produces a verifier-readable
  artifact or an explicit failure.

## What Remains In `nvidia-nat-harbor`

- NAT-backed Harbor agent wrapper and optional library-mode runner.
- Host-local NAT examples and runbooks.
- ATIF inline verifier and script bridge compatibility path.
- NAT evaluator dispatch and custom evaluator callable support.
- Simple calculator Harbor adapters and example E2E coverage.

These pieces stay out of Harbor core to preserve ownership boundaries and avoid
coupling Harbor to NAT evaluator internals.

## PR Sequence

### Phase A: Harbor Core

1. Add first-class `--env local` and local environment implementation.
2. Add generic local install policy primitives and setup metadata.
3. Land generic verifier extension hooks. Current side-branch status:
   `VerifierFactory`, import-path wiring, and verifier kwargs are implemented.
4. Decide agent run-mode ownership:
   - generic run-mode extension point, or
   - direct Harbor `nemo-agent` library-mode support.

### Phase B: `nvidia-nat-harbor`

1. Consume Harbor local env directly once `--env local` lands.
2. Consume Harbor verifier import hook with `ATIFInlineVerifier`.
3. Keep `bridge_runner` as a script compatibility path only.
4. Keep NAT evaluator dispatch and custom evaluator support in the package.
5. Document local env, shell mode, library mode, inline verifier, and script
   bridge as distinct modes.

### Phase C: Stabilization

Validate the full matrix before removing temporary workarounds:

- Shell compatibility mode with host Python wiring.
- Local env path mapping and artifact layout.
- Library-mode inline agent execution.
- Inline verifier with builtin trajectory evaluator.
- Inline verifier with builtin tunable-rag evaluator.
- Inline verifier with custom callable evaluator.
- Script bridge compatibility through task `tests/test.sh`.
- Concurrent inline trials with serialized environment overlays.
- No regression in existing Harbor shell-mode workflows.

## Final Acceptance Criteria

- Harbor supports `--env local` natively.
- Harbor exposes generic verifier import hooks and stable verifier contracts.
- Harbor has either a generic agent run-mode extension point or direct
  `nemo-agent` library-mode ownership.
- `nvidia-nat-harbor` no longer carries generic Harbor workarounds.
- `nvidia-nat-harbor` still owns NAT evaluator dispatch, ATIF sample coercion,
  custom callable evaluator support, and NAT-specific examples.
- All stabilization matrix lanes pass without patching installed packages inside
  `.venv` or `site-packages`.
