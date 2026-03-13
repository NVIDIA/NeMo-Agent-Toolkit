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

# User Identity Resolution

The NeMo Agent Toolkit automatically resolves a user identity from every incoming HTTP request and WebSocket connection. The server inspects standard credentials — session cookies, JWT Bearer tokens, API keys, and username and password headers — to determine who is making the request and assigns a stable, deterministic `user_id` that persists across sessions. This allows workflows to operate with a consistent view of the user without requiring workflow authors to handle credential parsing or identity logic directly.

## Overview

Identity resolution provides the following capabilities:

- **Automatic credential detection**: The server inspects incoming connections for known credential formats and resolves them into a user identity without any workflow-level configuration.
- **Deterministic user IDs**: Each credential produces a stable UUID v5. The same credential always resolves to the same `user_id`, regardless of transport (HTTP or WebSocket) or credential format.
- **Multiple credential types**: Session cookies, JWT Bearer tokens, opaque API keys, `X-API-Key` headers, HTTP Basic Auth, and WebSocket auth messages are all supported.
- **Per-user workflow support**: When a workflow is configured as per-user, the resolved `user_id` is used to isolate workflow state per user. Each user gets their own workflow instance.

:::{warning}
Identity resolution is an identity mapping step, not an authentication or authorization layer. **JSON Web Tokens are decoded with `verify_signature=False`** — the server trusts whatever credential arrives. The resolved `user_id` controls access to per-user workflow state (conversation history, builders, cached tokens). In production, deploy an authenticating reverse proxy or auth middleware that validates JSON Web Tokens before they reach NeMo Agent Toolkit. Without upstream verification, any party that can send HTTP requests to NeMo Agent Toolkit can impersonate any user. For credential validation, see [Authentication Providers](./api-authentication.md).
:::

## Supported Identity Sources

The following table lists all credential types that can be resolved into a user identity:

| **Source** | **Transport** | **How it arrives** |
|---|---|---|
| Session cookie | HTTP / WebSocket | `nat-session` cookie or `?session=` query parameter |
| JWT Bearer token | HTTP / WebSocket | `Authorization: Bearer <jwt>` header |
| API key (Bearer) | HTTP / WebSocket | `Authorization: Bearer <opaque-key>` header |
| API key (header) | HTTP / WebSocket | `X-API-Key: <key>` header |
| HTTP Basic Auth | HTTP / WebSocket | `Authorization: Basic <base64(user:pass)>` header |
| Auth message | WebSocket only | `auth_message` JSON payload (see [WebSocket Auth Message](../../reference/rest-api/websockets.md#auth-message)) |

## How It Works

Each request or connection should include exactly one credential. The server detects the credential type automatically and resolves it into a user identity. If multiple credential types are present, the server uses the first one it finds and ignores the rest.

For WebSocket connections, credentials can be provided either at connect time (via headers or cookies) or after the connection is established by sending an `auth_message`. When an `auth_message` is used, the resolved `user_id` is persisted for the duration of the session and applied to all subsequent workflow requests on that connection.

If no credential is found, the request proceeds without a user identity (anonymous).

:::{note}
For per-user workflows, a valid identity is required. If no credential can be resolved, the server returns an error instructing the client to provide a valid `Authorization` header or send an `auth_message`.
:::

## User ID Derivation

Each identity source produces a deterministic UUID v5 using a toolkit-specific namespace. The identity key varies by credential type:

| **Identity Source** | **Identity Key** | **Standards Reference** |
|---|---|---|
| JWT | First non-empty value from `sub`, `email`, `preferred_username` | [RFC 7519 Section 4.1.2](https://www.rfc-editor.org/rfc/rfc7519#section-4.1.2), [OpenID Connect Core 1.0 Section 5.1](https://openid.net/specs/openid-connect-core-1_0.html#StandardClaims) |
| API key | Raw API key string | — |
| Basic Auth | `base64(username:password)` ¹ | [RFC 7617](https://www.rfc-editor.org/rfc/rfc7617) |
| Session cookie | Raw cookie value | — |

¹ Because the password is part of the identity key, changing a password produces a new `user_id`. The user's prior per-user workflow state (conversation history, builders) becomes inaccessible.

**JWT claim precedence**: The `sub` claim is preferred as the stable, locally-unique subject identifier per RFC 7519. If `sub` is absent or empty, the resolver falls back to `email` and then `preferred_username` as defined by OpenID Connect Core 1.0 Standard Claims. If none of these claims contain a usable value, the server rejects the token with an error.

## Related Documentation

- [WebSocket Message Schema](../../reference/rest-api/websockets.md) — WebSocket message types including `auth_message` and `auth_response_message`
