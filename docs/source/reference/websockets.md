<!--
SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
This document defines the schema for WebSocket messages exchanged between the client and the AIQ toolkit server. Its primary
purpose is to guide users on how to interact with the AIQ toolkit server via WebSocket connection. Users can reliably
send and receive data while ensuring compatibility with the web server’s expected format. Additionally, this schema
provides flexibility for users to build and customize their own user interface by defining how different message types
should be handled, displayed, and processed. With a clear understanding of the message structure, developers can
seamlessly integrate their customized user interfaces with the AIQ toolkit server.

## Overview
The message schema described below facilitates transactional interactions with the AIQ toolkit server. The messages follow a
structured JSON format to ensure consistency in communication and can be categorized into two main types: `User Messages`
and `System Messages`. User messages are sent from the client to the server. System messages are sent from the server
to the client.

## Explanation of Fields
- `type`: Defines the category of the message.
    - Possible values:
      - `user_message`
      - `system_intermediate_message`
      - `system_response_message`
      - `system_interaction_message`
      - `user_interaction_message`
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
- `security`: Stores security information such as `api_key`, auth token etc. - OPTIONAL
    -   `api_key`: API key
    - token: auth or access token
- `error`: error information object
- `schema_version`: schema version - `OPTIONAL`

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
  "security": {
    "api_key": "string",
    "token": "string"
  },
  "error": {
    "code": "string",
    "message": "string",
    "details": "object"
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
  "content": {
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Yes continue processing sensitive information"
          },
        ]
      }
    ]
  },
  "timestamp": "string",
  "user": {
    "name": "string",
    "email": "string"
  },
  "security": {
    "api_key": "string",
    "token": "string"
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
  "content": {
    "text": "Response token can be json, code block or plain text"
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:02Z"
}
```

### System Response Token Message, Type: `error_message`
Definition: This message sends various types of error content to the client.
#### System Response Token Message Error Type Example:
```json
{
  "type": "error_message",
  "id": "token_001",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "content": {
      "code": "111", "message": "ValidationError", "details": "The provided email format is invalid."
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:02Z"
}

```
## System Human Interaction Message
System Human Interaction messages are sent from the server to the client containing Human Prompt content.

### Text Input Interaction
#### Text Input Interaction Message Example:
```json
{
  "type": "system_interaction_message",
  "id": "interaction_303",
  "thread_id": "thread_456",
  "parent_id": "id from user message",
  "content": {
      "input_type": "text",
      "text": "Hello, how are you today?",
      "placeholder": "Ask anything.",
      "required": true
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
      "required": true
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Multiple Choice Interaction, Type: `radio`
#### Radio Multiple Choice Interaction Example:
```json
{
  "type": "system_human_interaction",
  "id": "interaction_305",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
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
    "required": true
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Multiple Choice Interaction, Type: `checkbox`
#### Checkbox Multiple Choice Interaction Example:
```json
{
  "type": "system_human_interaction_name",
  "id": "interaction_306",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
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
    "required": true
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```

### Multiple Choice Interaction, Type: `dropdown`
#### Dropdown Multiple Choice Interaction Example:
```json
{
  "type": "system_human_interaction",
  "id": "interaction_305",
  "thread_id": "thread_456",
  "parent_id": "msg_123",
  "content": {
    "interaction": "system_human_interaction_name",
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
    "required": true
  },
  "status": "in_progress",
  "timestamp": "2025-01-13T10:00:03Z"
}
```
