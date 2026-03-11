<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# WebSocket Message Schema
This document defines the schema for WebSocket messages exchanged between the client and the NeMo Agent Toolkit server. Its primary
purpose is to guide users on how to interact with the NeMo Agent Toolkit server via WebSocket connection. Users can reliably
send and receive data while ensuring compatibility with the web server's expected format. Additionally, this schema
provides flexibility for users to build and customize their own user interface by defining how different message types
should be handled, displayed, and processed. With a clear understanding of the message structure, developers can
seamlessly integrate their customized user interfaces with the NeMo Agent Toolkit server.

## Overview
The message schema described below facilitates transactional interactions with the NeMo Agent Toolkit server. The messages follow a
structured JSON format to ensure consistency in communication and can be categorized into two main types: `User Messages`
and `System Messages`. User messages are sent from the client to the server. System messages are sent from the server
to the client.

**Client to server:**

- `auth_message` — Authentication credentials (JWT, API key, or username and password).
- `user_interaction_message` — Response to a human-in-the-loop prompt.
- `user_message` — Text or multimodal content from the user.

**Server to client:**

- `auth_response_message` — Result of an authentication attempt.
- `error_message` — Error details from the server.
- `observability_trace_message` — Observability trace ID for request correlation.
- `system_intermediate_message` — Intermediate step output during workflow execution.
- `system_interaction_message` — Human-in-the-loop prompt sent to the client.
- `system_response_message` — Final response content from the workflow.

## Auth Message

This message allows clients to authenticate over a WebSocket connection when header-based or
cookie-based authentication is not feasible (e.g., browser WebSocket APIs that do not support custom headers).
The server validates the credentials, resolves a user identity, and associates it with the current session.
The server responds with an `auth_response_message` in both cases — with `status: "success"` and the resolved
`user_id` on success, or `status: "error"` with structured error details on failure.

### Auth Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"auth_message"`. |
| `payload` | yes | object | Authentication credentials. Must include a `method` field: `"jwt"` (with `token`), `"api_key"` (with `token`), or `"basic"` (with `username`, `password`). |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |

### JWT Auth Message Example:
```json
{
  "type": "auth_message",
  "payload": { "method": "jwt", "token": "<jwt-token>" }
}
```

### API Key Auth Message Example:
```json
{
  "type": "auth_message",
  "payload": { "method": "api_key", "token": "<api-key>" }
}
```

### Basic Auth Message Example:
```json
{
  "type": "auth_message",
  "payload": { "method": "basic",
               "username": "<username>",
               "password": "<password>" }
}
```

## Auth Response Message

The server responds to an `auth_message` with an `auth_response_message` indicating success (with the resolved
`user_id`) or failure (with structured error details).

### Auth Response Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"auth_response_message"`. |
| `payload` | no | object | Error details (`code`, `message`, `details`). Present on failure, `null` on success. |
| `status` | yes | string | Outcome of the authentication attempt. One of `"success"`, `"error"`. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |
| `user_id` | no | string | Resolved user identifier. Present on success, `null` on failure. |

### Auth Success Response Example:
```json
{
  "type": "auth_response_message",
  "status": "success",
  "user_id": "5a3f8e2b-1c4d-5e6f-7a8b-9c0d1e2f3a4b",
  "payload": null,
  "timestamp": "2025-01-13T10:00:00Z"
}
```

### Auth Failure Response Example:
```json
{
  "type": "auth_response_message",
  "status": "error",
  "user_id": null,
  "payload": { "code": "user_auth_error",
               "message": "Authentication failed",
               "details": "Could not resolve user identity from auth payload (method=jwt)" },
  "timestamp": "2025-01-13T10:00:00Z"
}
```

## Error Message

This message sends various types of error content to the client. The `content` object matches the Error model:
`code` is one of `unknown_error`, `workflow_error`, `invalid_message`, `invalid_message_type`,
`invalid_user_message_content`, `invalid_data_content`, `user_auth_error`; `message` and `details` are strings.

### Error Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"error_message"`. |
| `content` | yes | object | Error object with `code`, `message`, and `details`. |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `id` | no | string | Message identifier. |
| `parent_id` | no | string | Links to the originating user message. |
| `status` | yes | string | Processing state. One of `"in_progress"`, `"complete"`. |
| `thread_id` | no | string | Thread identifier. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |

### Error Message Example:
```json
{
  "type": "error_message",
  "id": "token_001",
  "thread_id": "thread_456",
  "parent_id": "msg_001",
  "conversation_id": "conv_abc123",
  "content": { "code": "workflow_error",
               "message": "Invalid email format.",
               "details": "ValidationError" },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:02Z"
}
```

## Observability Trace Message

This message contains the observability trace ID for tracking requests across services.

### Observability Trace Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"observability_trace_message"`. |
| `content` | yes | object | Trace content with `observability_trace_id`. |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `id` | no | string | Message identifier. |
| `parent_id` | no | string | Links to the originating user message. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |

### Observability Trace Message Example:
```json
{
  "type": "observability_trace_message",
  "id": "trace_001",
  "parent_id": "msg_001",
  "conversation_id": "conv_abc123",
  "content": { "observability_trace_id": "019a9f4d-072a-77b0-aff1-262550329c13" },
  "timestamp": "2025-01-20T10:00:00Z"
}
```

## System Intermediate Step Message

This message contains the intermediate step content from a running workflow.

### System Intermediate Step Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"system_intermediate_message"`. |
| `content` | yes | object | Intermediate step content (`name` and `payload`). |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `id` | no | string | Message identifier. |
| `intermediate_parent_id` | no | string | Links to the parent intermediate step. |
| `parent_id` | no | string | Links to the originating user message. |
| `status` | yes | string | Processing state. One of `"in_progress"`, `"complete"`. |
| `thread_id` | no | string | Thread identifier. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |
| `update_message_id` | no | string | Identifier of the message being updated. |

### System Intermediate Step Message Example:
```json
{
  "type": "system_intermediate_message",
  "id": "step_789",
  "thread_id": "thread_456",
  "parent_id": "msg_001",
  "intermediate_parent_id": "step_788",
  "update_message_id": "step_789",
  "conversation_id": "conv_abc123",
  "content": { "name": "Query rephrasing",
               "payload": "Rephrased: What were the Q3 2025 revenue figures?" },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:01Z"
}
```

## System Interaction Message

System Interaction messages are sent from the server to the client containing Human Prompt content.

Each interaction prompt `content` object supports the following optional fields:

- `timeout`: Timeout in seconds for the prompt. Defaults to `null` (no timeout). When set, the frontend should display
  a countdown timer. If the user does not respond within the specified duration, the frontend should dismiss the prompt
  and display the `error` message. The server also enforces this timeout and raises a `TimeoutError` to the workflow.
  The value is set per-prompt by the workflow code. See the
  [Interactive Workflows Guide](../../build-workflows/advanced/interactive-workflows.md) for details.
- `error`: Error message to display on the prompt if the timeout expires or another error occurs. Defaults to
  `"This prompt is no longer available."`.

The `content.input_type` field determines the prompt format: `text`, `notification`, `binary_choice`, `radio`,
`checkbox`, or `dropdown`.

### System Interaction Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"system_interaction_message"`. |
| `content` | yes | object | Human-in-the-loop prompt content. |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `id` | no | string | Message identifier. |
| `parent_id` | no | string | Links to the originating user message. |
| `status` | yes | string | Processing state. One of `"in_progress"`, `"complete"`. |
| `thread_id` | no | string | Thread identifier. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |

### Text Input Interaction Example (Default, No Timeout):
```json
{
  "type": "system_interaction_message",
  "id": "interaction_303",
  "thread_id": "thread_456",
  "parent_id": "msg_001",
  "conversation_id": "conv_abc123",
  "content": {
      "input_type": "text",
      "text": "Hello, how are you today?",
      "placeholder": "Ask anything.",
      "required": true,
      "timeout": null,
      "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Text Input Interaction Example (With Timeout Configured):
```json
{
  "type": "system_interaction_message",
  "id": "interaction_303",
  "thread_id": "thread_456",
  "parent_id": "msg_001",
  "conversation_id": "conv_abc123",
  "content": {
      "input_type": "text",
      "text": "Hello, how are you today?",
      "placeholder": "Ask anything.",
      "required": true,
      "timeout": 300,
      "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Binary Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_304",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "conv_abc123",
  "content": {
    "input_type": "binary_choice",
    "text": "Should I continue or cancel?",
    "options": [
      { "id": "continue", "label": "Continue", "value": "continue" },
      { "id": "cancel", "label": "Cancel", "value": "cancel" }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Radio Multiple Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_305",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "conv_abc123",
  "content": {
    "input_type": "radio",
    "text": "Please select your preferred notification method:",
    "options": [
      { "id": "email",
        "label": "Email",
        "value": "email",
        "description": "Receive notifications via email" },
      { "id": "sms",
        "label": "SMS",
        "value": "sms",
        "description": "Receive notifications via SMS" },
      { "id": "push",
        "label": "Push Notification",
        "value": "push",
        "description": "Receive notifications via push" }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Checkbox Multiple Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_306",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "conv_abc123",
  "content": {
    "input_type": "checkbox",
    "text": "Select all notification methods you'd like to enable:",
    "options": [
      { "id": "email",
        "label": "Email",
        "value": "email",
        "description": "Receive notifications via email" },
      { "id": "sms",
        "label": "SMS",
        "value": "sms",
        "description": "Receive notifications via SMS" },
      { "id": "push",
        "label": "Push Notification",
        "value": "push",
        "description": "Receive notifications via push" }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Dropdown Multiple Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_307",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "conv_abc123",
  "content": {
    "input_type": "dropdown",
    "text": "Please select your preferred notification method:",
    "options": [
      { "id": "email",
        "label": "Email",
        "value": "email",
        "description": "Receive notifications via email" },
      { "id": "sms",
        "label": "SMS",
        "value": "sms",
        "description": "Receive notifications via SMS" },
      { "id": "push",
        "label": "Push Notification",
        "value": "push",
        "description": "Receive notifications via push" }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

## System Response Message

This message contains the final response content from a running workflow.

### System Response Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"system_response_message"`. |
| `content` | yes | object | Final response content (`text`). |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `id` | no | string | Message identifier. |
| `parent_id` | no | string | Links to the originating user message. |
| `status` | yes | string | Processing state. One of `"in_progress"`, `"complete"`. |
| `thread_id` | no | string | Thread identifier. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |

### System Response Message Example:
```json
{
  "type": "system_response_message",
  "id": "token_001",
  "thread_id": "thread_456",
  "parent_id": "msg_001",
  "conversation_id": "conv_abc123",
  "content": { "text": "The quarterly revenue was $4.2M, a 15% increase over Q2." },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:02Z"
}
```

## User Interaction Message

This message contains the response content from the human in the loop interaction.

### User Interaction Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"user_interaction_message"`. |
| `content` | yes | object | Response content from the human-in-the-loop interaction. |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `error` | no | object | Error object (`code`, `message`, `details`). |
| `id` | no | string | Message identifier. |
| `parent_id` | no | string | Links to the originating message. |
| `schema_version` | no | string | Schema version. |
| `thread_id` | no | string | Thread identifier for the interaction. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |
| `user` | no | object | User information (`name`, `email`, etc.). |

### User Interaction Message Example:
```json
{
  "type": "user_interaction_message",
  "id": "interaction_resp_001",
  "thread_id": "thread_456",
  "parent_id": "interaction_303",
  "conversation_id": "conv_abc123",
  "content": {
    "messages": [
      { "role": "user", "content": [{ "type": "text", "text": "Yes, continue processing." }] }
    ]
  },
  "error": { "code": "unknown_error", "message": "", "details": "" },
  "timestamp": "2025-01-13T10:00:04Z",
  "user": { "name": "Alice", "email": "alice@example.com" },
  "schema_version": "1.0.0"
}
```

## User Message

This message sends text or multimodal content from the client to the server. The `content` field carries the full chat history between the user and assistant.

### User Message Fields

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `type` | yes | string | Message type identifier. Must be `"user_message"`. |
| `content` | yes | object | Message body containing the chat history. Supports OpenAI compatible chat objects (text, image, audio, streaming). |
| `conversation_id` | no | string | Groups messages within the same conversation. |
| `error` | no | object | Error object (`code`, `message`, `details`). |
| `id` | no | string | Message identifier. |
| `schema_type` | yes | string | Defines the response schema for the workflow. One of `"generate_stream"`, `"chat_stream"`, `"generate"`, `"chat"`. |
| `schema_version` | no | string | Schema version. |
| `timestamp` | no | string | ISO 8601 timestamp. Auto-generated if omitted. |
| `user` | no | object | User information (`name`, `email`, etc.). |

### User Message Example:
```json
{
  "type": "user_message",
  "schema_type": "chat_stream",
  "id": "msg_001",
  "conversation_id": "conv_abc123",
  "content": {
    "messages": [
      { "role": "user", "content": [{ "type": "text", "text": "Hello, how are you?" }] },
      { "role": "assistant", "content": [{ "type": "text", "text": "I'm doing well, thanks!" }] },
      { "role": "user", "content": [{ "type": "text", "text": "What were Q3 2025 revenues?" }] }
    ]
  },
  "error": { "code": "unknown_error", "message": "", "details": "" },
  "timestamp": "2025-01-13T10:00:00Z",
  "user": { "name": "Alice", "email": "alice@example.com" },
  "schema_version": "1.0.0"
}
```
