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

# Hermes NeMo-Flow Harbor Smoke

This developer workflow runs Hermes Agent on one Harbor SWE-bench task with the
NeMo-Flow CLI gateway enabled. It validates the first available non-patching
Hermes instrumentation path before the upstream Hermes native middleware path
is available.

The validation pass currently relies on the gateway-emitted ATIF artifact. PR
#88 adds native ATOF JSONL exporter APIs to NeMo-Flow, but the Hermes CLI
gateway path still does not expose a raw ATOF JSONL artifact from `nemo-flow
run`/`hook-forward`. For this first SWE-bench smoke, skip the ATOF-derived
validation path and compare the native Hermes ATIF directly against the
gateway-emitted NeMo-Flow ATIF.

## TODO: Config-Based ATOF Export

Follow up once NeMo-Flow exposes configuration-based CLI exporter registration
for the gateway, for example through a project/user/global `plugin.toml` or
`plugins.toml` file depending on the final schema. At that point, update this
smoke to declare an ATOF exporter in the NeMo-Flow gateway configuration,
require `agent/nemo-flow-atof/events.jsonl`, convert it to
`agent/nemo-flow-atof-atif/trajectory.json`, and compare that ATOF-derived ATIF
against the native Hermes ATIF. Until that lands, absence of
`agent/nemo-flow-atof/events.jsonl` is expected.

## Pipeline

```mermaid
flowchart TD
  task[SWE-bench task<br/>django__django-13741] --> harbor[Harbor trial<br/>Docker task environment]

  harbor --> run[HermesNeMoFlow run<br/>nemo-flow CLI gateway wrapper]
  run --> nativeSession[agent/hermes-session.jsonl<br/>Hermes native session export]
  nativeSession --> nativeAtif[agent/trajectory.json<br/>Harbor Hermes ATIF]

  run --> gatewayHooks[Hermes hooks + nemo-flow gateway]
  gatewayHooks -. future CLI config .-> atof[agent/nemo-flow-atof/events.jsonl<br/>raw ATOF JSONL]
  atof -. deferred .-> converter[NAT ATOF-to-ATIF converter]
  converter -. deferred .-> nfAtif[agent/nemo-flow-atof-atif/trajectory.json<br/>ATOF-derived ATIF]
  gatewayHooks --> gatewayAtif[agent/nemo-flow-gateway-atif/trajectory.json<br/>canonical gateway-emitted ATIF]

  harbor --> swe[SWE-bench verifier<br/>patch + task tests]
  swe --> result[result.json<br/>reward/resolved]

  nativeAtif --> compare[Trajectory comparison<br/>tool sequence + NAT evaluators + Phoenix]
  gatewayAtif --> compare
  result --> outcome[Outcome sanity check<br/>successful/resolved run]
```

One Hermes run emits two trajectory artifacts today, with a third deferred:

- Native path: Hermes session export -> Harbor Hermes adapter ->
  `agent/trajectory.json`. This is Harbor's native Hermes trajectory and is
  produced independently of the NeMo-Flow gateway.
- Gateway-emitted ATIF: Hermes hooks and routed model traffic -> NeMo-Flow CLI
  gateway -> `agent/nemo-flow-gateway-atif/trajectory.json`. This is the
  canonical NeMo-Flow trajectory artifact and is the primary comparison target.
- ATOF-derived ATIF (TODO): once NeMo-Flow supports configuration-based ATOF
  exporter registration for CLI gateway sessions, raw events should land at
  `agent/nemo-flow-atof/events.jsonl` and the NAT ATOF-to-ATIF converter should
  populate `agent/nemo-flow-atof-atif/trajectory.json`. This path is not part
  of the current first-sample validation.

## Prerequisites

- Docker is running.
- NAT is checked out to a branch containing
  `nat_harbor.agents.installed.hermes_nemoflow:HermesNeMoFlow`.
- Harbor is installed from the source branch used by the Harbor integration.
- NeMo-Flow is checked out to a revision that contains the renamed
  `crates/cli` crate and PR #88. The published binary is `nemo-flow`, the
  published Cargo package is `nemo-flow-cli`, and Hermes gateway ATIF export
  uses `--atif-dir`.

<!-- path-check-skip-begin -->
```bash
mkdir -p external

if [ ! -d external/harbor/.git ]; then
  git clone https://github.com/AnuradhaKaruppiah/harbor.git external/harbor
fi
git -C external/harbor fetch origin
git -C external/harbor checkout ak-harbor-libary-mode

if [ ! -d external/nemo-flow/.git ]; then
  git clone https://github.com/NVIDIA/NeMo-Flow.git external/nemo-flow
fi
git -C external/nemo-flow fetch origin
git -C external/nemo-flow checkout main
```

The SWE-bench smoke task should exist at:

```text
external/harbor/datasets/swebench-opencode-smoke/django__django-13741
```

If that task is missing, create it with Harbor's SWE-bench adapter:

```bash
cd external/harbor/adapters/swebench

uv run swebench \
  --instance-id django__django-13741 \
  --task-dir ../../datasets/swebench-opencode-smoke \
  --overwrite

cd ../../../..
```

Use editable installs for local iteration:

```bash
uv venv --python 3.13 --seed .venv
uv pip install -e packages/nvidia_nat_harbor
uv pip install -e external/harbor
```
<!-- path-check-skip-end -->

`NVIDIA_BASE_URL` should point at the OpenAI-compatible NVIDIA endpoint used
for this smoke.

```bash
export NVIDIA_BASE_URL=<openai-compatible-nvidia-base-url>
```

## Run the Smoke

Create a local env file for the Docker task environment. Do not commit this
file.

<!-- path-check-skip-begin -->
```bash
mkdir -p .tmp/harbor/secrets
read -rsp 'NVIDIA_API_KEY: ' NVIDIA_API_KEY; echo
cat > .tmp/harbor/secrets/nvidia.env <<EOF
NVIDIA_API_KEY=${NVIDIA_API_KEY}
NVIDIA_BASE_URL=${NVIDIA_BASE_URL}
EOF
```

Run the NeMo-Flow-enabled Hermes smoke:

```bash
export HARBOR_JOBS_DIR=.tmp/harbor/hermes-nemoflow-smoke
export SWEBENCH_TASK=external/harbor/datasets/swebench-opencode-smoke/django__django-13741
export NEMO_FLOW_REPO="$PWD/external/nemo-flow"
export JOB_NAME=hermes-nemoflow-repeatable-smoke-1

set -a
. .tmp/harbor/secrets/nvidia.env
set +a

.venv/bin/harbor run \
  --path "$SWEBENCH_TASK" \
  -l 1 \
  --job-name "$JOB_NAME" \
  --jobs-dir "$HARBOR_JOBS_DIR" \
  --yes -n 1 --max-retries 0 \
  --env-file .tmp/harbor/secrets/nvidia.env \
  --agent-import-path nat_harbor.agents.installed.hermes_nemoflow:HermesNeMoFlow \
  --env docker \
  --model nvidia/opus-frontier \
  --ak nemo_flow_repo="$NEMO_FLOW_REPO" \
  --ak fail_missing_nemoflow_atof=false \
  --ak fail_missing_nemoflow_atif=true
```

Expected artifacts under the trial directory (today, against current
NeMo-Flow main):

```text
agent/hermes.txt
agent/hermes-session.jsonl
agent/trajectory.json
agent/nemo-flow-gateway-atif/trajectory.json
result.json
verifier/report.json
```

After the config-based NeMo-Flow CLI ATOF exporter follow-up, these artifacts
should also appear:

```text
agent/nemo-flow-atof/events.jsonl
agent/nemo-flow-atof-atif/trajectory.json
```

## Quick Artifact Check

Set `TRIAL` to the completed trial directory:

```bash
export HARBOR_JOBS_DIR=.tmp/harbor/hermes-nemoflow-smoke
export JOB_NAME=hermes-nemoflow-repeatable-smoke-1
export TRIAL
TRIAL=$(find "$HARBOR_JOBS_DIR/$JOB_NAME" -maxdepth 1 -type d -name 'django__django-13741__*' | head -n 1)
test -n "$TRIAL"
```

Check that both ATIF artifacts load and expose token totals:

```bash
.venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

from nat_harbor.verifier.evaluator_adapter import load_atif_samples

trial = Path(os.environ["TRIAL"])
agent = trial / "agent"
required = (
    "hermes.txt",
    "hermes-session.jsonl",
    "trajectory.json",
    "nemo-flow-gateway-atif/trajectory.json",
)
optional = (
    "nemo-flow-atof/events.jsonl",
    "nemo-flow-atof-atif/trajectory.json",
)
for rel in required:
    path = agent / rel
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    print("ok", rel, path.stat().st_size)
for rel in optional:
    path = agent / rel
    if path.exists():
        print("ok (optional)", rel, path.stat().st_size)
    else:
        print("absent (deferred)", rel)

for rel in ("trajectory.json", "nemo-flow-gateway-atif/trajectory.json"):
    samples = load_atif_samples(agent / rel)
    trajectory = samples[0].trajectory
    data = json.loads((agent / rel).read_text())
    metrics = data.get("final_metrics") or {}
    print(rel, trajectory.schema_version, len(trajectory.steps), metrics)

atof_path = agent / "nemo-flow-atof/events.jsonl"
if atof_path.exists():
    print("atof_events", sum(1 for _ in atof_path.open()))
PY
```

Compare the native and gateway-emitted tool sequences:

```bash
.venv/bin/python -m nat_harbor.smoke.compare_atif_tools \
  --native "$TRIAL/agent/trajectory.json" \
  --candidate "$TRIAL/agent/nemo-flow-gateway-atif/trajectory.json"
```

The deterministic comparison should be `match (same)` or `match (richer)` for a
healthy gateway-emitted trajectory. Do not require the deferred ATOF-derived
comparison for this first SWE-bench sample.

## Post-Run Trajectory Scoring

Run the scorer without LLM calls first:

```bash
.venv/bin/python -m nat_harbor.smoke.score_atif_trajectories \
  --job-dir "$HARBOR_JOBS_DIR/$JOB_NAME" \
  --candidate-rel agent/nemo-flow-gateway-atif/trajectory.json \
  --output-dir "$HARBOR_JOBS_DIR/$JOB_NAME/post-run-scores" \
  --no-llm
```

Run the LLM scoring pass with an OpenAI-compatible judge endpoint:

```bash
set -a
. .tmp/harbor/secrets/nvidia.env
set +a

export OPENAI_API_KEY="$NVIDIA_API_KEY"
export OPENAI_BASE_URL="$NVIDIA_BASE_URL"
export NAT_HARBOR_TRAJECTORY_JUDGE_MODEL=<openai-compatible-judge-model>

.venv/bin/python -m nat_harbor.smoke.score_atif_trajectories \
  --job-dir "$HARBOR_JOBS_DIR/$JOB_NAME" \
  --candidate-rel agent/nemo-flow-gateway-atif/trajectory.json \
  --output-dir "$HARBOR_JOBS_DIR/$JOB_NAME/post-run-llm-scores" \
  --config-file packages/nvidia_nat_harbor/configs/opencode-nemoflow-trajectory-eval.yml \
  --evaluator-name trajectory_eval \
  --score-timeout-sec 45
```

## Phoenix Inspection

If Phoenix is running locally at `http://localhost:6006`, export the two ATIF
artifacts to separate projects:

```bash
ENDPOINT=http://localhost:6006/v1/traces

.venv/bin/python -m nat.plugins.phoenix.scripts.export_trajectory_to_phoenix.export_atif_trajectory_to_phoenix \
  "$TRIAL/agent/trajectory.json" \
  --endpoint "$ENDPOINT" \
  --project harbor-hermes-native

.venv/bin/python -m nat.plugins.phoenix.scripts.export_trajectory_to_phoenix.export_atif_trajectory_to_phoenix \
  "$TRIAL/agent/nemo-flow-gateway-atif/trajectory.json" \
  --endpoint "$ENDPOINT" \
  --project harbor-hermes-nemoflow-gateway
```

Open `http://localhost:6006` and compare the two projects.

## Known Limitations

- The wrapper builds `nemo-flow-cli` (binary `nemo-flow`) inside each Harbor
  task container, so the first run is slow.
- PR #88 adds native ATOF JSONL exporter APIs, but the Hermes CLI gateway path
  does not yet expose a raw ATOF JSONL artifact through configuration. The
  wrapper keeps the ATOF code path best-effort
  (`fail_missing_nemoflow_atof=false`), so today's runs validate the
  gateway-emitted ATIF and skip the ATOF-derived comparison performed by the
  OpenCode smoke. TODO: when NeMo-Flow lands configuration-based CLI ATOF
  exporter registration, add that gateway config to this smoke and flip
  `fail_missing_nemoflow_atof=true` to re-enable strict validation.
- Complete LLM lifecycle telemetry requires Hermes model traffic to use the
  NeMo-Flow gateway. This wrapper configures that path for `nvidia`, `openai`,
  `openrouter`, and `anthropic` model prefixes.
- The upstream Hermes native middleware path remains a later comparison lane.
