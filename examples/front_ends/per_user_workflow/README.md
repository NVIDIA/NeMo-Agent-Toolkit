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
# Per-User Workflow Example

This example demonstrates the **per-user workflow pattern** in NeMo Agent toolkit. With this pattern, each user gets their own isolated workflow and function instances with separate state.

## Overview

The per-user workflow pattern is useful when you need:
- **User-isolated state**: Each user's data is completely separate from other users
- **Stateful functions**: Functions that maintain state across requests for the same user
- **Session-based personalization**: User preferences, history, or context that persists within a session

## Components

### Per-User Functions

1. **`per_user_notepad`**: A simple notepad that stores notes per user
   - Each user has their own list of notes
   - Notes added by one user are not visible to other users

2. **`per_user_preferences`**: A preferences store per user
   - Each user has their own preference settings
   - Changes by one user don't affect other users

### Per-User Workflow

**`per_user_assistant`**: A workflow that combines the notepad and preferences functions
- Tracks session statistics per user
- Provides a unified command interface

## Usage

### 1. Install the Example

First, install the example package:

```bash
uv pip install -e ./examples/front_ends/per_user_workflow
```

### 2. Start the Server

```bash
nat serve --config_file=examples/front_ends/per_user_workflow/configs/config.yml
```

**expected output**
```console
% nat serve --config_file=examples/front_ends/per_user_workflow/configs/config.yml
2025-12-08 11:20:09 - INFO     - nat.cli.commands.start:192 - Starting NAT from config file: 'examples/front_ends/per_user_workflow/configs/config.yml'
2025-12-08 11:20:12 - INFO     - nat.front_ends.fastapi.fastapi_front_end_plugin:138 - Created local Dask cluster with scheduler at tcp://127.0.0.1:58705 using processes workers
WARNING:  Current configuration will not reload as not all conditions are met, please refer to documentation.
INFO:     Started server process [23491]
INFO:     Waiting for application startup.
2025-12-08 11:20:13 - INFO     - nat.front_ends.fastapi.fastapi_front_end_plugin_worker:245 - No evaluators configured, skipping evaluator initialization
2025-12-08 11:20:13 - INFO     - nat.runtime.session:266 - Workflow is per-user (entry_function=None)
2025-12-08 11:20:13 - INFO     - nat.front_ends.fastapi.fastapi_front_end_plugin_worker:724 - Expecting generate request payloads in the following format: {'command': FieldInfo(annotation=str, required=True, description="Command to execute: 'note', 'pref', 'stats', or 'help'"), 'action': FieldInfo(annotation=str, required=False, default='', description='Action for the command'), 'param1': FieldInfo(annotation=str, required=False, default='', description='First parameter (key/content)'), 'param2': FieldInfo(annotation=str, required=False, default='', description='Second parameter (value)')}
2025-12-08 11:20:13 - INFO     - nat.runtime.session:266 - Workflow is per-user (entry_function=None)
2025-12-08 11:20:13 - INFO     - nat.runtime.session:266 - Workflow is per-user (entry_function=None)
2025-12-08 11:20:13 - INFO     - nat.front_ends.fastapi.fastapi_front_end_plugin_worker:592 - Added evaluate_item route at /evaluate/item
```

### 2. Test with Different Users

Each user is identified by the `nat-session` cookie. Different session IDs represent different users. Run the following
commands in a separate terminal.

#### User 1 Operations

```bash
# Add a note as User 1
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "note", "action": "add", "param1": "Alices first note"}'
```

**Expected Output**
```console
{"success":true,"message":"Note added successfully","data":{"notes":[],"count":1,"commands_executed":1}}
```

# List notes as User 1
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "note", "action": "list"}'
```

**Expected Output**
```console
{"success":true,"message":"Found 1 notes","data":{"notes":["Alices first note"],"count":1,"commands_executed":2}}
```

#### User 2 Operations

```bash
# List notes as User 2 (should be empty - isolated from User 1)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d '{"command": "note", "action": "list"}'
```

**Expected Output**
```console
{"success":true,"message":"Found 0 notes","data":{"notes":[],"count":0,"commands_executed":1}}
```

# Add a note as User 2
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d '{"command": "note", "action": "add", "param1": "Bobs note"}'
```

**Expected Output**
```console
{"success":true,"message":"Note added successfully","data":{"notes":[],"count":1,"commands_executed":2}}
```

#### Preferences

```bash
# Set a preference as User 1
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "pref", "action": "set", "param1": "theme", "param2": "light"}'
```

**Expected Output**
```console
{"success":true,"message":"Preference 'theme' set to 'light'","data":{"value":"","preferences":{"theme":"light","language":"en","notifications":"enabled"},"commands_executed":1}}
```

# Check User 2's theme (should still be "dark" from defaults)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=bob" \
  -d '{"command": "pref", "action": "get", "param1": "theme"}'
```

**Expected Output**
```console
{"success":true,"message":"Preference 'theme' = 'dark'","data":{"value":"dark","preferences":{},"commands_executed":1}}
```

#### Help and Stats

```bash
# Get help
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "help"}'
```

**Expected Output**
```console
{"success":true,"message":"Session statistics: 1 commands executed","data":{"commands_executed":1}}%
```

# Get session stats (tracks commands per user)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Cookie: nat-session=alice" \
  -d '{"command": "stats"}'
```

**Expected Output**
```console
{"success":true,"message":"Session statistics: 2 commands executed","data":{"commands_executed":2}}%
```

## Available Commands

| Command | Action | Parameters | Description |
|---------|--------|------------|-------------|
| `note` | `add` | `param1`: content | Add a note |
| `note` | `list` | - | List all notes |
| `note` | `clear` | - | Clear all notes |
| `note` | `count` | - | Count notes |
| `pref` | `set` | `param1`: key, `param2`: value | Set a preference |
| `pref` | `get` | `param1`: key | Get a preference |
| `pref` | `list` | - | List all preferences |
| `help` | - | - | Show help message |
| `stats` | - | - | Show session statistics |

## Configuration

The `config.yml` file configures:

- **`per_user_workflow_timeout`**: How long inactive user sessions are kept (default: 30 minutes)
- **`per_user_workflow_cleanup_interval`**: How often to check for inactive sessions (default: 5 minutes)
- **`max_notes`**: Maximum notes per user (default: 50)
- **`default_preferences`**: Default preferences for new users

## How It Works

1. **User Identification**: Users are identified by the `nat-session` cookie
2. **On-Demand Creation**: Per-user workflow builders are created when a user first makes a request
3. **State Isolation**: Each user's functions maintain separate state
4. **Automatic Cleanup**: Inactive user sessions are automatically cleaned up based on the configured timeout
