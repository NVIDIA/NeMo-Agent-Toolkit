<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scripts for ATIF Evaluation Utilities

## Convert `workflow_output.json` to `workflow_output_atif.json`

Use `convert_workflow_output_to_atif.py` to convert legacy IST workflow dumps into ATIF sample output format.

### Script

- `packages/nvidia_nat_eval/scripts/convert_workflow_output_to_atif.py`

### Example

```bash
python packages/nvidia_nat_eval/scripts/convert_workflow_output_to_atif.py \
  --input ".tmp/nat/examples/advanced_agents/alert_triage_agent/output/offline_atif/workflow_output.json" \
  --output-dir ".tmp/nat/examples/advanced_agents/alert_triage_agent/output/offline_atif"
```

This writes:

- `.tmp/nat/examples/advanced_agents/alert_triage_agent/output/offline_atif/workflow_output_atif.json`

### Required input setting

To make conversion complete, the IST dump must include all events.

- Set `workflow_output_step_filter: []` in your eval config (empty list means no filtering).
- If event filtering is enabled, required IST events can be missing and conversion output can be incomplete or incorrect.

For example, in:

- `examples/advanced_agents/alert_triage_agent/configs/config_offline_atif.yml`

keep:

```yaml
workflow_output_step_filter: []
```
