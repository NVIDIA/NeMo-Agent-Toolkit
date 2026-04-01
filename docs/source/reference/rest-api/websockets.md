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
send and receive data while ensuring compatibility with the web server’s expected format. Additionally, this schema
provides flexibility for users to build and customize their own user interface by defining how different message types
should be handled, displayed, and processed. With a clear understanding of the message structure, developers can
seamlessly integrate their customized user interfaces with the NeMo Agent Toolkit server.

## Overview
The message schema described below facilitates transactional interactions with the NeMo Agent Toolkit server. The messages follow a
structured JSON format to ensure consistency in communication and can be categorized into two main types: `User Messages`
and `System Messages`. User messages are sent from the client to the server. System messages are sent from the server
to the client.

## Explanation of Fields
- `type`: Defines the category of the message.
    - Possible values:
      - `auth_message`
      - `auth_response_message`
      - `user_message`
      - `system_intermediate_message`
      - `system_response_message`
      - `system_interaction_message`
      - `user_interaction_message`
      - `observability_trace_message`
      - `error_message`
- `schema_type`:  Defines the response schema for a given workflow
- `id`: A unique identifier for the message.
    - Purpose: Used for tracking, referencing, and updating messages.
- `conversation_id`: A unique identifier used to associate all messages and interactions with a specific conversation session.
    - Purpose: Groups-related messages within the same conversation/chat feed.
- `parent_id`: Links a message to its originating message.
    -   Optional: Used for responses, updates, or continuations of earlier messages.
- `content`: Stores the main data of the message.
    - Format: String for text messages and array for contents which can have attachments such as image, audio and videos. See above example.
    -   Attachments support OpenAI compatible chat objects such as (Default, Image, Audio, and Streaming)
- `status`: Indicates the processing state of the message.
    - Possible values: `in_progress`, `completed`, `failed`.
    - Optional: Typically used for system messages.
- `timestamp`: Captures when the message was created or updated.
     - Format: ISO 8601 (e.g., `2025-01-13T10:00:00Z`).
 - `user`: Stores user information - OPTIONAL
    -   name: User name
    -   email: User email
    -   other info: Any other information
- `error`: Error information object with `code` (string, see Error types), `message` (string), and `details` (string)
- `schema_version`: schema version - `OPTIONAL`

## Auth Message
This message allows clients to authenticate over a WebSocket connection when header-based or
cookie-based authentication is not feasible (e.g., browser WebSocket APIs that do not support custom headers).
The server validates the credentials, resolves a user identity, and associates it with the current session.
The server responds with an `auth_response_message` in both cases — with `status: "success"` and the resolved
`user_id` on success, or `status: "error"` with structured error details on failure.

### JWT Auth Message Example:
```json
{
  "type": "auth_message",
  "payload": {
    "method": "jwt",
    "token": "<jwt-token>"
  }
}
```

### API Key Auth Message Example:
```json
{
  "type": "auth_message",
  "payload": {
    "method": "api_key",
    "token": "<api-key>"
  }
}
```

### Basic Auth Message Example:
```json
{
  "type": "auth_message",
  "payload": {
    "method": "basic",
    "username": "<username>",
    "password": "<password>"
  }
}
```

## Auth Response Message
The server responds to an `auth_message` with an `auth_response_message` indicating success (with the resolved
`user_id`) or failure (with structured error details).

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
  "payload": {
    "code": "user_auth_error",
    "message": "Authentication failed",
    "details": "Could not resolve user identity from auth payload (method=jwt)"
  },
  "timestamp": "2025-01-13T10:00:00Z"
}
```

## User Message Examples
### User Message - (OpenAI compatible)
Definition: This message is used to send text content to a running workflow. The entire chat history between the user
and assistant is persisted in the message history and only the last `user` message in the list will be processed by the
running workflow.

#### User Message Example:
```json
{
  "type": "user_message",
  "schema_type": "string",
  "id": "string",
  "conversation_id": "string",
  "content": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Hello, how are you?"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "im good"
          }
        ]
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "solve this question"
          }
        ]
      }
    ]
  },
  "timestamp": "string",
  "user": {
    "name": "string",
    "email": "string"
  },
  "error": {
    "code": "string",
    "message": "string",
    "details": "string"
  },
  "schema_version": "string"
}
```

### User Interaction Message - (OpenAI compatible)
Definition: This message contains the response content from the human in the loop interaction.

#### User Interaction Message Example:
```json
{
  "type": "user_interaction_message",
  "id": "string",
  "thread_id": "string",
  "parent_id": "string",
  "conversation_id": "string",
  "content": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Yes continue processing sensitive information"
          }
        ]
      }
    ]
  },
  "timestamp": "string",
  "user": {
    "name": "string",
    "email": "string"
  },
  "schema_version": "string"
}
```

## System Message Examples
### System Intermediate Step Message
Definition: This message contains the intermediate step content from a running workflow.
#### System Intermediate Step Message Example:
```json
{
  "type": "system_intermediate_message",
  "id": "step_789",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "intermediate_parent_id": "default",
  "conversation_id": "string",
  "content": {
    "name": "name of the step - example Query rephrasal",
    "payload": "Step information, it can be json or code block or it can be plain text"
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:01Z"
}
```

### System Response Token Message, Type: `system_response_message`
Definition: This message contains the final response content from a running workflow.
#### System Response Token Message Example

```json
{
  "type": "system_response_message",
  "id": "token_001",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "conversation_id": "string",
  "content": {
    "text": "Response token can be json, code block or plain text"
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:02Z"
}
```

### System Response Token Message, Type: `error_message`
Definition: This message sends various types of error content to the client. The `content` object matches the Error model: `code` is one of `unknown_error`, `workflow_error`, `invalid_message`, `invalid_message_type`, `invalid_user_message_content`, `invalid_data_content`, `user_auth_error`; `message` and `details` are strings.
#### System Response Token Message Error Type Example:
```json
{
  "type": "error_message",
  "id": "token_001",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "conversation_id": "string",
  "content": {
    "code": "workflow_error",
    "message": "The provided email format is invalid.",
    "details": "ValidationError"
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:02Z"
}
```
## System Human Interaction Message
System Human Interaction messages are sent from the server to the client containing Human Prompt content.

Each interaction prompt `content` object supports the following optional fields:

- `timeout`: Timeout in seconds for the prompt. Defaults to `null` (no timeout). When set, the frontend should display
  a countdown timer. If the user does not respond within the specified duration, the frontend should dismiss the prompt
  and display the `error` message. The server also enforces this timeout and raises a `TimeoutError` to the workflow.
  The value is set per-prompt by the workflow code. See the
  [Interactive Workflows Guide](../../build-workflows/advanced/interactive-workflows.md) for details.
- `error`: Error message to display on the prompt if the timeout expires or another error occurs. Defaults to
  `"This prompt is no longer available."`.

### Text Input Interaction
#### Text Input Interaction Message Example (Default, No Timeout):
```json
{
  "type": "system_interaction_message",
  "id": "interaction_303",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "conversation_id": "string",
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

#### Text Input Interaction Message Example (With Timeout Configured):
```json
{
  "type": "system_interaction_message",
  "id": "interaction_303",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "conversation_id": "string",
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
### Binary Choice Interaction (Yes/No, Continue/Cancel)
#### Binary Choice Interaction Message Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_304",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "string",
  "content": {
      "input_type": "binary_choice",
      "text": "Should I continue or cancel?",
      "options": [{
          "id": "continue",
          "label": "Continue",
          "value": "continue",
      }, {
          "id": "cancel",
          "label": "Cancel",
          "value": "cancel",
      }],
      "required": true,
      "timeout": null,
      "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Multiple Choice Interaction, Type: `radio`
#### Radio Multiple Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_305",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "string",
  "content": {
    "input_type": "radio",
    "text": "I'll send you updates about the analysis progress. Please select your preferred notification method:",
    "options": [
      {
        "id": "email",
        "label": "Email",
        "value": "email",
        "description": "Receive notifications via email"
      },
      {
        "id": "sms",
        "label": "SMS",
        "value": "sms",
        "description": "Receive notifications via SMS"
      },
      {
        "id": "push",
        "label": "Push Notification",
        "value": "push",
        "description": "Receive notifications via push"
      }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Multiple Choice Interaction, Type: `checkbox`
#### Checkbox Multiple Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_306",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "string",
  "content": {
    "input_type": "checkbox",
    "text": "The analysis will take approximately 30 minutes to complete. Select all notification methods you'd like to enable:",
    "options": [
      {
        "id": "email",
        "label": "Email",
        "value": "email",
        "description": "Receive notifications via email"
      },
      {
        "id": "sms",
        "label": "SMS",
        "value": "sms",
        "description": "Receive notifications via SMS"
      },
      {
        "id": "push",
        "label": "Push Notification",
        "value": "push",
        "description": "Receive notifications via push"
      }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Multiple Choice Interaction, Type: `dropdown`
#### Dropdown Multiple Choice Interaction Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_307",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "conversation_id": "string",
  "content": {
    "input_type": "dropdown",
    "text": "I'll send you updates about the analysis progress. Please select your preferred notification method:",
    "options": [
      {
        "id": "email",
        "label": "Email",
        "value": "email",
        "description": "Receive notifications via email"
      },
      {
        "id": "sms",
        "label": "SMS",
        "value": "sms",
        "description": "Receive notifications via SMS"
      },
      {
        "id": "push",
        "label": "Push Notification",
        "value": "push",
        "description": "Receive notifications via push"
      }
    ],
    "required": true,
    "timeout": null,
    "error": "This prompt is no longer available."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### System Observability Trace Message
Definition: This message contains the observability trace ID for tracking requests across services.
#### System Observability Trace Message Example:
```json
{
  "type": "observability_trace_message",
  "id": "trace_001",
  "parent_id": "id from user message",
  "conversation_id": "string",
  "content": {
    "observability_trace_id": "019a9f4d-072a-77b0-aff1-262550329c13"
  },
  "timestamp": "2025-01-20T10:00:00Z"
}
```
