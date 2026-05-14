<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A365 Troubleshooting

## 401 / 403 During Telemetry Export

Start with these checks:

1. confirm the token audience is correct for the target API
2. confirm the expected identity is being used for the worker lane
3. confirm the tenant has granted the required permissions

Common values to verify in your environment:

- tenant ID
- observability audience

## Teams Works In Web Chat But Not In Teams

Check:

1. Teams manifest `bots[].botId`
2. Azure Bot Microsoft App ID
3. `A365_ALLOWED_AUDIENCES`
4. `A365_APP_ID`

The worker lane can legitimately use the blueprint ID as `A365_APP_ID` while
accepting the bot app ID as an allowed audience for inbound Teams traffic.

## Graph Mail MCP Is Unreachable

Check:

1. `GRAPH_MAIL_MCP_URL`
2. the Graph mail MCP service is running and reachable
3. `GRAPH_MAIL_TOKEN` if your deployment requires delegated Graph auth at the MCP layer
4. worker logs for MCP registration or connection failures

## Notifications

Notification handlers exist in the plugin code, but the example keeps
`enable_notifications: false` in the primary deployment config.

Reason:

- the current `microsoft-agents-a365-notifications` package path was observed
  to be unstable enough that the safest demo shape keeps notifications off

Do not claim notification routing as part of the primary validated example
until that path is re-tested end to end.

If you do test the included
[../configs/a365_email_notifications.yml](../configs/a365_email_notifications.yml)
lane, verify these separately:

1. the `microsoft-agents-a365-notifications` package is installed and importable
2. the worker starts with `enable_notifications: true`
3. the notification source is actually delivering email events to the runtime
4. the default or configured `notification_workflow` is reachable

## Layered Validation

Validate failures in this order:

1. route health at `/api/messages`
2. workflow execution
3. MCP tool registration
4. telemetry ingestion

It is normal for these layers to fail independently. A healthy route does not
prove Teams works, and a working workflow does not prove telemetry is accepted.
